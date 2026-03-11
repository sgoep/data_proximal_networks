import torch
from src.radon import RadonAdapter

@torch.no_grad()
def landweber(
    A,
    AT,
    g: torch.Tensor,
    x0: torch.Tensor,
    omega: float,
    n_iter: int,
) -> torch.Tensor:
    """
    Landweber iterations for min_x 0.5||A x - g||^2.

    Args:
        A: operator with forward(x) and backward(y)
        g: measurements (B,1,angles,detectors)
        x0: initial image (B,1,H,W)
        omega: step size, should satisfy 0 < omega < 2/||A||^2
        n_iter: number of iterations
    """
    x = x0.clone()
    for _ in range(n_iter):
        r = g - A(x)      # residual in sinogram space
        x = x + omega * AT(r)
    return x