import torch.nn as nn
from src.radon import RadonAdapter
import torch
from src.total_variation import tv_cp


class RESNET(nn.Module):
    """
    Simple residual wrapper around a UNet.

    The network learns a residual correction which is added to the input.
    """

    def __init__(self, unet: nn.Module):
        super().__init__()
        self.unet = unet

    def forward(self, x, y_delta=None):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image.
        y_delta : torch.Tensor, optional
            Unused (kept for interface compatibility).

        Returns
        -------
        torch.Tensor
            Residual-enhanced output x + UNet(x).
        """
        res = self.unet(x)
        return x + res


class NSN(nn.Module):
    """
    Null-Space Network (NSN).

    Learns corrections that live in the null space of the Radon operator
    by projecting UNet outputs onto unmeasured angles and backprojecting.
    """

    def __init__(self, unet: nn.Module, radon: RadonAdapter):
        super().__init__()
        self.unet = unet
        self.radon = radon

    def forward(self, x, y_delta=None):
        """
        Forward pass applying null-space correction.

        Parameters
        ----------
        x : torch.Tensor
            Input image.
        y_delta : torch.Tensor, optional
            Unused (kept for interface compatibility).

        Returns
        -------
        torch.Tensor
            Input image plus null-space correction.
        """
        res = self.unet(x)
        x_nsn = self.radon.fbp(
            self.radon.proj_nsn(self.radon.forward(res))
        )
        return x + x_nsn


class DPNSN(nn.Module):
    """
    Data-Proximal Null-Space Network (DP-NSN).

    Combines:
      - data-consistent correction on measured angles (L2-ball projection)
      - null-space correction on unmeasured angles
    """

    def __init__(self, unet: nn.Module, radon: RadonAdapter, beta: float):
        super().__init__()
        self.unet = unet
        self.radon = radon
        self.beta = beta

    @staticmethod
    def _proj_l2_ball(v: torch.Tensor, radius: float) -> torch.Tensor:
        """
        Project batch of tensors onto an L2 ball of given radius.
        """
        B = v.shape[0]
        n = torch.linalg.norm(v.view(B, -1), dim=1).clamp_min(1e-12)
        scale = torch.minimum(torch.ones_like(n), (radius / n)).view(B, 1, 1, 1)
        return v * scale

    def forward(self, x, y_delta=None):
        """
        Forward pass with data projection and null-space correction.

        Parameters
        ----------
        x : torch.Tensor
            Input image.
        y_delta : torch.Tensor, optional
            Unused (kept for interface compatibility).

        Returns
        -------
        torch.Tensor
            Corrected reconstruction.
        """
        res = self.unet(x)
        y = self.radon.forward(res)

        x_dp = self.radon.fbp_la(
            self._proj_l2_ball(self.radon.proj_ran(y), self.beta)
        )
        x_nsn = self.radon.fbp(self.radon.proj_nsn(y))

        return x + (x_dp + x_nsn)


class DPNSN_RES(nn.Module):
    """
    Residual Data-Proximal Null-Space Network.

    Uses measured-data residuals to enforce data consistency while
    learning null-space and residual corrections.
    """

    def __init__(self, unet: nn.Module, radon: RadonAdapter, beta: float):
        super().__init__()
        self.unet = unet
        self.radon = radon
        self.beta = beta 

    @staticmethod
    def _proj_l2_ball(v: torch.Tensor, radius: float) -> torch.Tensor:
        """
        Project batch of tensors onto an L2 ball of given radius.
        """
        B = v.shape[0]
        n = torch.linalg.norm(v.view(B, -1), dim=1).clamp_min(1e-12)
        scale = torch.minimum(torch.ones_like(n), (radius / n)).view(B, 1, 1, 1)
        return v * scale

    def forward(self, x: torch.Tensor, y_delta: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using residual sinogram consistency.

        Parameters
        ----------
        x : torch.Tensor
            Current reconstruction.
        y_delta : torch.Tensor
            Measured sinogram residual (measured angles).

        Returns
        -------
        torch.Tensor
            Updated reconstruction with residual and null-space corrections.
        """
        res = self.unet(x)

        y = self.radon.forward(res)
        r = self.radon.proj_ran(y - y_delta)
        r_ball = self._proj_l2_ball(r, self.beta)

        x_dp = res - self.radon.fbp_la(r_ball)
        x_nsn = self.radon.fbp(self.radon.proj_nsn(y))

        return x + x_dp + x_nsn
