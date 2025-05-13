# Code borrowed from https://github.com/envfluids/PanguWeather/tree/optim-dev-2 

import torch
from torch import nn


class PatchRecovery2D(nn.Module):
    """
    Patch Embedding Recovery to 2D Image.

    Args:
        img_size (tuple[int]): Lat, Lon
        patch_size (tuple[int]): Lat, Lon
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.ConvTranspose2d(in_chans, out_chans, patch_size, patch_size)

    def forward(self, x):
        output = self.conv(x)
        _, _, H, W = output.shape
        h_pad = H - self.img_size[0]
        w_pad = W - self.img_size[1]

        padding_top = h_pad // 2
        padding_bottom = int(h_pad - padding_top)

        padding_left = w_pad // 2
        padding_right = int(w_pad - padding_left)

        return output[:, :, padding_top: H - padding_bottom, padding_left: W - padding_right]


class PatchRecovery3D(nn.Module):
    """
    Patch Embedding Recovery to 3D Image.
    Note: here Pl becomes T for our data [B, C, T, X, Y]

    Args:
        img_size (tuple[int]): Pl, Lat, Lon
        patch_size (tuple[int]): Pl, Lat, Lon
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.ConvTranspose3d(in_chans, out_chans, patch_size, patch_size)

        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor):
        output = self.conv(x)
        _, _, Pl, Lat, Lon = output.shape

        pl_pad = Pl - self.img_size[0]
        lat_pad = Lat - self.img_size[1]
        lon_pad = Lon - self.img_size[2]

        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front

        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top

        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

        return output[:, :, padding_front: Pl - padding_back,
               padding_top: Lat - padding_bottom, padding_left: Lon - padding_right]


# borrowed from
#https://gist.github.com/A03ki/2305398458cb8e2155e8e81333f0a965
def ICNR(tensor, initializer, upscale_factor=2, *args, **kwargs):
    "tensor: the 2-dimensional Tensor or more"
    upscale_factor_squared = upscale_factor * upscale_factor
    assert tensor.shape[0] % upscale_factor_squared == 0, \
        ("The size of the first dimension: "
         f"tensor.shape[0] = {tensor.shape[0]}"
         " is not divisible by square of upscale_factor: "
         f"upscale_factor = {upscale_factor}")
    sub_kernel = torch.empty(tensor.shape[0] // upscale_factor_squared,
                             *tensor.shape[1:])
    sub_kernel = initializer(sub_kernel, *args, **kwargs)
    return sub_kernel.repeat_interleave(upscale_factor_squared, dim=0)

class SubPixelConvICNR_2D(nn.Module):
    """
    Patch Embedding Recovery to 2D Image.

    Args:
        img_size (tuple[int]): Lat, Lon
        patch_size (tuple[int]): Lat, Lon
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.img_size = img_size
        assert patch_size[0] == patch_size[1], 'mismatch'
        self.conv = nn.Conv2d(in_chans, out_chans*patch_size[0]**2, kernel_size=3, stride=1, padding=1, bias=0)
        self.pixelshuffle = nn.PixelShuffle(patch_size[0])
        weight = ICNR(self.conv.weight,
                      initializer=nn.init.kaiming_normal_,
                      upscale_factor=patch_size[0])
        self.conv.weight.data.copy_(weight)   # initialize conv.weight

    def forward(self, x):
        output = self.conv(x)

        output = self.pixelshuffle(output)

        _, _, H, W = output.shape
        h_pad = H - self.img_size[0]
        w_pad = W - self.img_size[1]

        padding_top = h_pad // 2
        padding_bottom = int(h_pad - padding_top)

        padding_left = w_pad // 2
        padding_right = int(w_pad - padding_left)

        return output[:, :, padding_top: H - padding_bottom, padding_left: W - padding_right]


class SubPixelConvICNR_3D(nn.Module):
    """
    Patch Embedding Recovery to 3D Image.

    Args:
        img_size (tuple[int]): Pl, Lat, Lon
        patch_size (tuple[int]): Pl, Lat, Lon
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.img_size = img_size
        self.patch_size_t = patch_size[0]
        assert patch_size[1] == patch_size[2], 'mismatch'
        self.in_chans_per_frame = in_chans // patch_size[0]  # patch_size is tubelet size
        self.conv = nn.Conv2d(self.in_chans_per_frame, out_chans*patch_size[1]**2, kernel_size=3, stride=1, padding=1, bias=0, padding_mode='circular')
        self.pixelshuffle = nn.PixelShuffle(patch_size[1])
        weight = ICNR(self.conv.weight,
                      initializer=nn.init.kaiming_normal_,
                      upscale_factor=patch_size[1])
        self.conv.weight.data.copy_(weight)   # initialize conv.weight

    def forward(self, x: torch.Tensor):
        # first, split in dimension
        # print(x.shape)
        x = x.reshape(x.shape[0], self.in_chans_per_frame, self.patch_size_t, *x.shape[2:]).flatten(2, 3)[:, :, 0:self.img_size[0]] # to make 13 vertical dims
        #print(x.shape)
        x = x.movedim(-3, 1).flatten(0, 1)
        output = self.conv(x)
        output = self.pixelshuffle(output)
        #print(output.shape)
        output = output.reshape(-1, self.img_size[0], *output.shape[1:]).movedim(1, -3)

        _, _, Pl, Lat, Lon = output.shape

        pl_pad = Pl - self.img_size[0]
        lat_pad = Lat - self.img_size[1]
        lon_pad = Lon - self.img_size[2]

        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front

        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top

        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

        return output[:, :, padding_front: Pl - padding_back,
               padding_top: Lat - padding_bottom, padding_left: Lon - padding_right]
