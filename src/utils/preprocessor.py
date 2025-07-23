import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.signal.windows import gaussian, tukey


def get_spectral_preprocessor(params):
    if params.preprocess == 'FourierPatch':
        return FourierPatchFilterPreprocess(params)
    elif params.preprocess == 'Fourier':
      return FourierFilterPreprocessor(params)
    elif params.preprocess == 'Wavelet':
      raise NotImplementedError
      #return WaveletFilterPreprocessor(params)
    else:
      raise ValueError("Invalid preprocessor type.")


class SpatialLayerNorm(nn.Module):
    """
    Apply a per-sample normalization to mitigate spatial signal 
    amplitude discrepancies across batch members as a result of
    random spectral filtering.
    """
    def __init__(self, num_channels, affine=False):
        super().__init__()
        self.affine = affine
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1, 1))
        else:
            self.register_buffer('gamma', torch.ones(1, num_channels, 1, 1, 1))
            self.register_buffer('beta', torch.ones(1, num_channels, 1, 1, 1))

    def forward(self, x):
        # x.shape: (B, C, T, H, W)
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        std = x.std(dim=(2, 3, 4), keepdim=True) + 1e-6
        return self.gamma * (x - mean) / std + self.beta
    

class FourierFilterPreprocessor(nn.Module):
    """
    Preprocessor that spectrally filters data along spatial dimensions
    using Fourier transform.
    """

    def __init__(self, params):
        super().__init__()

        self.filter_size = params['img_size']
        self.window_width = params['window_width']   # between (0, img_size/2)
        self.window_center_kx = params['window_center_kx']   # range for locating center of mask (lower, upper)
        self.window_center_ky = params['window_center_ky']
        self.window_type = params['window_type']   # 'rectangular', 'gaussian' or 'tukey'
        self.window_gaussian_std = params['window_gaussian_std']
        self.window_tukey_alpha = params['window_tukey_alpha']
        self.randomized_filters = params['randomized_filters']   # randomly generate filters along batch dim
        self.filter_shuffle = params['filter_shuffle']   # Shuffle f1/f2 randomly along batch dim
        self.use_spectral_weights = params['use_spectral_weights']
        self.spectrally_weigh_input = params['spectrally_weigh_input']
        self.spectrally_weigh_output = params['spectrally_weigh_output']

        self.apply_norm = params["spectral_mask_apply_norm"]  # bool

        if self.apply_norm:
            self.norm = SpatialLayerNorm(
                num_channels = params["in_chans"],
                affine = params["spectral_mask_apply_affine"]
            )
        else:
            self.norm = nn.Identity()

        self.enable_curriculum_learning = params['enable_curriculum_learning']  # bool
        self.cl_epochs = params['cl_epochs'] # list of epochs for cl
        self.cl_kx_values = params['cl_window_centers_kx']  # list of kx values (vs epoch) for cl
        self.cl_ky_values = params['cl_window_centers_ky']  # list of ky values (vs epoch) for cl

    def _create_window(self):
        """Create a 2D window for frequency domain filtering."""

        # Create 1D window
        if self.window_type == 'gaussian':
            window = gaussian(self.window_width, std=self.window_gaussian_std)
        elif self.window_type == 'tukey':
            window = tukey(self.window_width, alpha=self.window_tukey_alpha)
        elif self.window_type == 'rectangular':
            window = tukey(self.window_width, alpha=0.)
        else:
            raise ValueError('Invalid window_type')
        
        return window

    def _create_filter_kernel(self, shifts):
        """
        Create bandpass filter kernel based on specified window type and cutoff.
        Args:
          shifts (tuple): (shift_x, shift_y)
        """

        window_x = self._create_window()
        
        window_x = torch.from_numpy(window_x).float().to(self.device)
        window_y = window_x.clone()

        # Pad to half filter size length, shift, then extend by mirroring
        window_x = F.pad(window_x, (0, self.filter_size//2 - self.window_width))
        window_x = torch.roll(window_x, shifts[0])
        window_x = torch.cat((window_x, torch.flipud(window_x)))
        window_y = F.pad(window_y, (0, self.filter_size//2 - self.window_width))
        window_y = torch.roll(window_y, shifts[1])
        window_y = torch.cat((window_y, torch.flipud(window_y)))

        window = torch.outer(window_x, window_y)
        return window, 1 - window

    def _interpolate_filter_centers(self):
        """
        Interpolate between filter centers (kx, ky) for given epoch.
        """
        if self.epoch <= self.cl_epochs[0]:
            return self.cl_kx_values[0], self.cl_ky_values[0]
        elif self.epoch >= self.cl_epochs[-1]:
            return self.cl_kx_values[-1], self.cl_ky_values[-1]
        else:
            # Find interval for linear interpolation
            for i in range(len(self.cl_epochs) - 1):
                if self.cl_epochs[i] <= self.epoch < self.cl_epochs[i+1]:
                    fraction = (self.epoch - self.cl_epochs[i]) / (self.cl_epochs[i+1] - self.cl_epochs[i])
                    kx = self.cl_kx_values[i] + fraction * (self.cl_kx_values[i+1] - self.cl_kx_values[i])
                    ky = self.cl_ky_values[i] + fraction * (self.cl_ky_values[i+1] - self.cl_ky_values[i])
                    return kx, ky

    def _generate_filter_batch(self):

        if self.enable_curriculum_learning:
            # Interpolate filter centers based on current epoch
            kx, ky = self._interpolate_filter_centers()
            # Create single filter
            shift_x, shift_y = int(kx - self.window_width//2), int(ky - self.window_width//2)
            f1, f2 = self._create_filter_kernel((shift_x, shift_y))
            # Expand into batch shape
            f1 = f1.unsqueeze(0).unsqueeze(0).unsqueeze(0)   # shape (1, 1, 1, h, w)
            f2 = f2.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            f1 = f1.expand(self.x_shape)
            f2 = f2.expand(self.x_shape)
        else:
            if not self.randomized_filters:
                # Create single filter
                shift_x, shift_y = self.window_center_kx[0] - self.window_width//2, self.window_center_ky[0] - self.window_width//2
                f1, f2 = self._create_filter_kernel((shift_x, shift_y))
                # Expand into batch shape
                f1 = f1.unsqueeze(0).unsqueeze(0).unsqueeze(0)   # shape (1, 1, 1, h, w)
                f2 = f2.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                f1 = f1.expand(self.x_shape)
                f2 = f2.expand(self.x_shape)
            else:
                f1 = torch.zeros(self.x_shape, device=self.device)
                f2 = torch.zeros(self.x_shape, device=self.device)
                for i in range(self.x_shape[0]):
                    # Select random cutoff between upper and lower thresholds
                    shift_x = torch.randint(low=self.window_center_kx[0], 
                                            high=self.window_center_kx[1], 
                                            size=(1,), 
                                            dtype=torch.int16) - self.window_width//2
                    shift_y = torch.randint(low=self.window_center_ky[0], 
                                            high=self.window_center_ky[0], 
                                            size=(1,), 
                                            dtype=torch.int16) - self.window_width//2
                    f1[i,..., :, :], f2[i, ..., :, :] = self._create_filter_kernel((shift_x, shift_y))

        self.example_filter = f1[0]

        # Swap f1/f2 in half of the batch
        if self.filter_shuffle:
            half_b = self.x_shape[0] // 2
            
            # Corrected version
            new_tophalf = f2[:half_b].clone()
            new_bottomhalf = f1[:half_b].clone()

            # Create temporary copies to avoid in-place modification issues
            temp_f1 = f1.clone()
            temp_f2 = f2.clone()

            # Perform the swap
            temp_f1[:half_b] = new_tophalf
            temp_f2[:half_b] = new_bottomhalf

            return temp_f1, temp_f2

        return f1, f2

    def _apply_filter(self, x, f1, f2):
        """Apply filter to data in spectral domain."""
        x_fft = torch.fft.fft2(x, dim=(-2, -1))  # 2D Fourier Transform
        x_fft_filtered_f1 = x_fft * f1
        x_fft_filtered_f2 = x_fft * f2

        x1 = torch.fft.ifft2(x_fft_filtered_f1, dim=(-2, -1)).real  # Inverse FFT
        x2 = torch.fft.ifft2(x_fft_filtered_f2, dim=(-2, -1)).real  # Inverse FFT

        # Compute weights
        # shape: [b, c, t] broadcasted to [b, c, t, x, y]
        x1_weights = torch.sum(torch.abs(x_fft)**2, dim=[-1, -2]) / torch.sum(torch.abs(x_fft_filtered_f1)**2, dim=[-1, -2])
        x1_weights = x1_weights.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.filter_size, self.filter_size)
        x2_weights = torch.sum(torch.abs(x_fft)**2, dim=[-1, -2]) / torch.sum(torch.abs(x_fft_filtered_f2)**2, dim=[-1, -2])
        x2_weights = x2_weights.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.filter_size, self.filter_size)

        if self.spectrally_weigh_input:
            x1 = x1 * x1_weights
        if self.spectrally_weigh_output:
            x2 = x2 * x2_weights

        # Applying layernorm only to input
        x1 = self.norm(x1)
        
        if self.use_spectral_weights:
            return x1, x2, x2_weights

        return x1, x2

    def _fftshift2d(self, x):
        """Apply fftshift to 2D image."""
        x = torch.fft.fftshift(torch.fft.fftshift(x, dim=[-2]), dim=[-1])
        return x

    def get_filter(self):
        """Get an example of a filter kernel - typically the
        filter appied to the first batch member."""
        return self._fftshift2d(self.example_filter)

    def forward(self, x, epoch):
        """
        Forward pass applying both filters (lowpass and highpass).
        Args:
          x (torch.Tensor): Input tensor of shape (batch, channels, time, height, width)
          epoch (optional (int)): current epoch for determining filter design in curriculum learning
        Returns:
          tuple: (x1, x2) filtered outputs
        """
        self.x_shape = x.shape
        self.device = x.device

        self.epoch = epoch
        
        # Create filters (lowpass and highpass)
        f1, f2 = self._generate_filter_batch()

        return self._apply_filter(x, f1, f2)


class FourierPatchFilterPreprocess(nn.Module):
    """
    Preprocessor that masks spectral patches in 3D Fourier domain.
    """

    def __init__(self, params):
        super().__init__()
        self.patch_size = params["spectral_patch_size"]               # (pt, ph, pw)
        self.mask_ratio = params["spectral_mask_ratio"]               # float in (0, 1)
        self.spatial_only = params["spectral_mask_spatial_only"]      # bool
        self.apply_norm = params["spectral_mask_apply_norm"]  # bool

        if self.apply_norm:
            self.norm = SpatialLayerNorm(
                num_channels = params["in_chans"],
                affine = params["spectral_mask_apply_affine"]
            )
        else:
            self.norm = nn.Identity()
            
        if self.spatial_only:
            self.fft_dims = (-2, -1)
        else:
            self.fft_dims = (-3, -2, -1)

        self.enable_curriculum_learning = params['enable_curriculum_learning']  # bool
        self.cl_epochs = params['cl_epochs'] # list of epochs for cl
        self.cl_mask_ratio_values = params['cl_mask_ratio_values']  # list of mask ratios (vs epoch) for cl

    def _create_positive_freq_mask(self, shape, device):
        b, c, t, h, w = shape
        pt, ph, pw = self.patch_size

        # For fftn (real input), we must include Nyquist freq
        t_half = t // 2 if t > 1 else 1
        h_half = h // 2
        w_half = w // 2

        if self.spatial_only:
            nt = 1
            pt = t  # single chunk of all time slices
            t_half = t
        else:
            nt = max(t_half // pt, 1) if t > 1 else 1
        nh = h_half // ph
        nw = w_half // pw

        total_patches = c * nt * nh * nw

        patch_mask = torch.ones((b, total_patches), device=device)
        for i in range(b):
            drop_count = int(self.mask_ratio * total_patches)
            drop_idx = torch.randperm(total_patches, device=device)[:drop_count]
            patch_mask[i, drop_idx] = 0.0

        patch_mask = patch_mask.view(b, c, nt, nh, nw)

        rep_t = pt if (not self.spatial_only and t > 1) else t

        freq_mask = patch_mask.repeat_interleave(rep_t, dim=2) \
                              .repeat_interleave(ph, dim=3) \
                              .repeat_interleave(pw, dim=4)

        # Trim to exact size
        freq_mask = freq_mask[:, :, :t_half, :h_half, :w_half]

        # Pad to include Nyquist freq (i.e., +1 along h and w)
        pad_t = (t % 2 == 0 and t > 1) and not self.spatial_only
        pad_h = (h % 2 == 0)
        pad_w = (w % 2 == 0)

        pad = [0, int(pad_w), 0, int(pad_h), 0, int(pad_t)]  # F.pad expects reversed order
        freq_mask = torch.nn.functional.pad(freq_mask, pad, mode='replicate')

        return freq_mask  # now shape: [b, c, t//2 (+1), h//2+1, w//2+1]

    def _mirror_hermitian(self, x, full_size, dim):
        """
        Reconstruct Hermitian-symmetric full mask along a given dim.
        Mirrors the positive frequencies excluding DC and Nyquist.
        """
        half = x.shape[dim]
        N = full_size[dim]
        if N == 1:
            return x

        flip_slice = [slice(None)] * x.ndim
        if N % 2 == 0:
            start = 1
            end = half - 1
        else:
            start = 1
            end = half

        # Only mirror if there's something to mirror
        if end <= start:
            return x

        flip_slice[dim] = slice(start, end)
        flipped = torch.flip(x[tuple(flip_slice)], dims=[dim])
        return torch.cat([x, flipped], dim=dim)

    def _make_hermitian_mask(self, half_mask, full_shape):

        mask = half_mask
        for dim in self.fft_dims:  
            mask = self._mirror_hermitian(mask, full_shape, dim)

        assert mask.shape[-(len(self.fft_dims)):] == full_shape[-(len(self.fft_dims)):], \
            f"Expected shape {full_shape[-(len(self.fft_dims)):]}, got {mask.shape[-(len(self.fft_dims)):]}"

        return mask
                
    def _update_mask_ratio(self):
        """
        Update mask ratio for curriculum learning.
        """

        if self.epoch <= self.cl_epochs[0]:
            self.mask_ratio = self.cl_mask_ratio_values[0]
        elif self.epoch >= self.cl_epochs[-1]:
            self.mask_ratio = self.cl_mask_ratio_values[-1]
        else:
            # Find interval for linear interpolation
            for i in range(len(self.cl_epochs) - 1):
                if self.cl_epochs[i] <= self.epoch < self.cl_epochs[i+1]:
                    fraction = (self.epoch - self.cl_epochs[i]) / (self.cl_epochs[i+1] - self.cl_epochs[i])
                    self.mask_ratio = self.cl_mask_ratio_values[i] + fraction * (self.cl_mask_ratio_values[i+1] - self.cl_mask_ratio_values[i])
                    break

    def forward(self, x, epoch):
        x_shape = x.shape
        device = x.device

        self.epoch = epoch
        if self.enable_curriculum_learning:
            self._update_mask_ratio()

        half_mask = self._create_positive_freq_mask(x_shape, device)
        full_mask = self._make_hermitian_mask(half_mask, x_shape)
        full_mask_complement = 1. - full_mask

        x_fft = torch.fft.fftn(x, dim=self.fft_dims, norm='ortho')

        x_fft_masked = x_fft * full_mask
        x_fft_masked_complement = x_fft * full_mask_complement

        x_out = torch.fft.ifftn(x_fft_masked, dim=self.fft_dims, norm='ortho').real
        x_out_complement = torch.fft.ifftn(x_fft_masked_complement, dim=self.fft_dims, norm='ortho').real

        # Applying layernorm only to input
        x_out = self.norm(x_out)

        return x_out, x_out_complement, full_mask