import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.signal.windows import gaussian, tukey


def get_spectral_preprocessor(params):
    if params.preprocess == 'Fourier':
      return FourierFilterPreprocessor(params)
    elif params.preprocess == 'Wavelet':
      raise NotImplementedError
      #return WaveletFilterPreprocessor(params)
    else:
      raise ValueError("Invalid preprocessor type.")


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

    def _create_window(self):
        """Create a 2D window for frequency domain filtering."""

        # Create 1D window
        if self.window_type == 'gaussian':
            window = gaussian(self.window_width, std=self.window_gaussian_std)
        elif self.window_type == 'tukey':
            window = tukey(self.window_width, alpha=self.window_tukey_alpha)
        elif self.window_type == 'rectangle':
            window = tukey(self.window_width, alpha=0.)
        
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

    def _generate_filter_batch(self):

        if not self.randomized_filters:
            # Create single filter
            shift_x, shift_y = self.window_center_kx[0] - self.mask_width//2, self.window_center_ky[0] - self.mask_width//2
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

    def forward(self, x):
        """
        Forward pass applying both filters (lowpass and highpass).
        Args:
          x (torch.Tensor): Input tensor of shape (batch, channels, time, height, width)
        Returns:
          tuple: (x1, x2) filtered outputs
        """
        self.x_shape = x.shape
        self.device = x.device

        # Create filters (lowpass and highpass)
        f1, f2 = self._generate_filter_batch()

        return self._apply_filter(x, f1, f2)