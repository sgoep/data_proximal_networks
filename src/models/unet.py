# %%
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore

from src.utils.load_config import load_config
from src.utils.radon_operator import filter_sinogram, get_radon_operators


def gaussian_kernel1d(sigma, size=5):
    """Creates a 1D Gaussian kernel."""
    x = torch.arange(-size // 2 + 1, size // 2 + 1).float()
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()  # Normalize
    return kernel


def chi_func(angles, phi, m, smooth=False, sigma=10, kernel_size=50):
    if isinstance(angles, np.ndarray):
        angles = torch.Tensor(angles)
    chi = torch.zeros_like(angles).cuda()
    chi[angles <= phi] = 1

    if smooth:
        # Create a Gaussian kernel
        kernel = gaussian_kernel1d(sigma, kernel_size).cuda()
        kernel = kernel.view(1, 1, -1)  # Reshape for 1D convolution

        # Reshape chi for convolution and apply the kernel
        chi = chi[None, None, :]  # Reshape chi to [batch, channels, width]
        chi = F.conv1d(chi, kernel, padding="same")
        chi = chi[0, 0, :]  # Remove added dimensions
        chi[0 : torch.argmax(chi)] = 1
    chi = chi[:, None]
    chi = torch.repeat_interleave(chi, m, dim=1).unsqueeze(0)
    # chi = torch.repeat_interleave(chi, n, dim=0).unsqueeze(1)
    return chi


class data_prox_func:
    def __init__(self, norm):
        self.beta = norm

    def forward(self, X):
        Y = torch.zeros_like(X).to("cuda")
        for j in range(Y.shape[0]):
            norm = torch.linalg.norm(X[j, 0, :, :])
            if norm <= self.beta:
                Y[j, 0, :, :] = X[j, 0, :, :]
            else:
                Y[j, 0, :, :] = self.beta * X[j, 0, :, :] / norm

        return Y


class projections:
    def __init__(self, config, smooth=False):
        self.chi = chi_func(
            config.angles_full, config.phi_limited, config.det_count, smooth=smooth
        ).to(config.device)

    def _proj(self, X):
        Y = torch.zeros_like(X).cuda()
        for j in range(X.shape[0]):
            Y[j, 0, :, :] = X[j, 0, :, :] * self.chi
        return Y

    def null_space_projection(self, X):
        return X - self._proj(X)

    def range_projection(self, X):
        return self._proj(X)


class DoubleConv(nn.Module):
    """
    A module that performs two consecutive convolution operations, each
    followed by a ReLU activation.

    This module implements a double convolution layer block, where each
    convolution is a 3x3 convolution followed by ReLU activation.

    Parameters:
    -----------
    in_channels : int
        The number of input channels.
    out_channels : int
        The number of output channels.
    mid_channels : int, optional
        The number of channels after the first convolution. If not specified,
        defaults to out_channels.

    Methods:
    --------
    forward(x: torch.Tensor) -> torch.Tensor
        Applies the double convolution block to the input tensor and returns
        the output.
    """

    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None
    ):

        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """
    A module that performs downsampling followed by a double convolution.

    This module first downsamples the input using max pooling and then applies
    a double convolution layer.

    Parameters:
    -----------
    in_channels : int
        The number of input channels.
    out_channels : int
        The number of output channels.

    Methods:
    --------
    forward(x: torch.Tensor) -> torch.Tensor
        Applies max pooling followed by a double convolution and returns
        the output.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    A module that performs upsampling followed by a double convolution.

    This module first upsamples the input using either bilinear upsampling or
    transposed convolution, and then applies a double convolution layer.

    Parameters:
    -----------
    in_channels : int
        The number of input channels.
    out_channels : int
        The number of output channels.
    bilinear : bool, optional
        Whether to use bilinear upsampling (True) or transposed convolution
        (False). Defaults to True.

    Methods:
    --------
    forward(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor
        Upsamples the input tensor `x1`, concatenates it with `x2`, and
        applies a double convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of
        # channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    A module that performs a 1x1 convolution to map the features to the
    desired number of output channels.

    Parameters:
    -----------
    in_channels : int
        The number of input channels.
    out_channels : int
        The number of output channels.

    Methods:
    --------
    forward(x: torch.Tensor) -> torch.Tensor
        Applies the 1x1 convolution and returns the output.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    A U-Net model with optional constraints and null-space regularization.

    This U-Net architecture supports additional constraints such as null-space
    projection and data-proximal projection using custom layers.

    Parameters:
    -----------
    n_channels : int
        The number of input channels.
    n_classes : int
        The number of output classes.
    constraint : bool
        Whether to apply a proximal operator constraint.
    null_space : bool
        Whether to apply null-space projection.
    norm2 : float, optional
        The ell2 norm threshold for the data-proximal projection. Required if
        `constraint` is True.
    bilinear : bool, optional
        Whether to use bilinear upsampling or transposed convolutions.
        Defaults to True.

    Attributes:
    -----------
    A : function
        A function representing the forward Radon transform.
    FBP : function
        A function representing the filtered backprojection (FBP) of the Radon
        transform.
    null_space_layer : null_space_layer, optional
        A layer that projects the input onto the null-space of the Radon
        transform.
    proximal_layer : proximal_layer, optional
        A layer that applies the proximal operator to the input.

    Methods:
    --------
    forward(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        Passes the input through the U-Net, applies optional constraints, and
        returns the output tensor.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        model_name,
        example,
        radon_full=None,
        smooth: bool = False,
        bilinear: bool = True,
    ):

        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        if True:
            if radon_full is None:
                radon_full = get_radon_operators(example)[0]
            config = load_config(example)
            self.config = config
            self.model_name = model_name
            self.radon_full = radon_full
            self.beta = config.norm
            self.data_prox_func = data_prox_func(config.norm)
            smooth = True if "smooth" in model_name else False
            self.projections = projections(config, smooth=smooth)

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = X

        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        res = self.outc(x)

        if True:
            if self.model_name in [
                "fbp_X_res",
                "tv_X_res",
                "ell1_X_res",
                "landweber_X_res",
            ]:
                # RESNet
                return x0 + res

            elif self.model_name.split("_")[-1] == "nsn":
                # NSN
                res_data = self.radon_full.forward(res)
                res_nsn = self.projections.null_space_projection(res_data)
                res = self.radon_full.backward(filter_sinogram(res_nsn))
                return x0 + res

            elif self.model_name.split("_")[-1] == "dp":
                # DP NSN
                res_data = self.radon_full.forward(res)
                res_ran = self.data_prox_func.forward(
                    self.projections.range_projection(res_data)
                )
                res_nsn = self.projections.null_space_projection(res_data)
                res = self.radon_full.backward(filter_sinogram(res_ran + res_nsn))
                return x0 + res

            else:
                raise ValueError(f"Unknown model {self.model_name}.")

        return res
