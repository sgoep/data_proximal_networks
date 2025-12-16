import torch.nn as nn
from src.radon import RadonAdapter
import torch
from src.total_variation import tv_cp

class RESNET(nn.Module):
    def __init__(self, unet: nn.Module):
        super().__init__()
        self.unet = unet

    def forward(self, x, y_delta=None):
        res = self.unet(x)
        return x + res

class NSN(nn.Module):
    def __init__(self, unet: nn.Module, radon: RadonAdapter):
        super().__init__()
        self.unet = unet
        self.radon = radon

    def forward(self, x, y_delta=None):
        res = self.unet(x)
        x_nsn = self.radon.fbp(self.radon.proj_nsn(self.radon.forward(res)))

        return x + x_nsn

class DPNSN(nn.Module):
    def __init__(self, unet: nn.Module, radon: RadonAdapter, beta: float):
        super().__init__()
        self.unet = unet
        self.radon = radon
        self.beta = beta

    # def dp_proj(self, y):
    #     norms = torch.linalg.norm(y.view(y.shape[0], -1), dim=1,)
    #     scale = torch.minimum(torch.ones_like(norms), self.beta / (norms + 1e-12),)
    #     scale = scale.view(-1, 1, 1, 1)
    #     return y * scale
    @staticmethod
    def _proj_l2_ball(v: torch.Tensor, radius: float) -> torch.Tensor:
        B = v.shape[0]
        n = torch.linalg.norm(v.view(B, -1), dim=1).clamp_min(1e-12)
        scale = torch.minimum(torch.ones_like(n), (radius / n)).view(B, 1, 1, 1)
        return v * scale

    def forward(self, x, y_delta=None):
        res = self.unet(x)

        y = self.radon.forward(res)
        # x_dp = self.radon.fbp(
        #     self.dp_proj(
        #         self.radon.proj_ran(y)
        #     )
        # )
        x_dp = self.radon.fbp_la(self._proj_l2_ball(self.radon.proj_ran(y), self.beta))
        x_nsn = self.radon.fbp(self.radon.proj_nsn(y))

        return x + (x_dp + x_nsn)
    
class DPNSN_RES(nn.Module):
    def __init__(self, unet: nn.Module, radon: RadonAdapter, beta: float):
        super().__init__()
        self.unet = unet
        self.radon = radon
        self.beta = beta

    @staticmethod
    def _proj_l2_ball(v: torch.Tensor, radius: float) -> torch.Tensor:
        B = v.shape[0]
        n = torch.linalg.norm(v.view(B, -1), dim=1).clamp_min(1e-12)
        scale = torch.minimum(torch.ones_like(n), (radius / n)).view(B, 1, 1, 1)
        return v * scale

    def forward(self, x: torch.Tensor, y_delta: torch.Tensor) -> torch.Tensor:
        res = self.unet(x)

        y = self.radon.forward(res)                 # full sinogram
        r = self.radon.proj_ran(y - y_delta)        # residual on measured angles
        r_ball = self._proj_l2_ball(r, self.beta)        # shrink residual to <= beta

        x_dp = res - self.radon.fbp_la(r_ball)           # preconditioned correction
        x_nsn = self.radon.fbp(self.radon.proj_nsn(y))

        return x + x_dp + x_nsn