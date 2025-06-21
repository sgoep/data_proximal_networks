from typing import Union

import torch
import torch_radon
from torch_radon.solvers import cg

from .my_shearlet import ShearletTransform


def shrink(a, b):
    """
    Applies a soft-thresholding (shrinkage) function for ℓ₁-regularization.

    This function is commonly used in optimization problems involving
    sparsity constraints, such as shearlet-based reconstruction.

    Args:
        a (torch.Tensor): Input tensor.
        b (torch.Tensor): Threshold value tensor.

    Returns:
        torch.Tensor: Shrunk tensor where values are reduced by `b` if they are larger than `b`.
    """
    return (torch.abs(a) - b).clamp_min(0) * torch.sgn(a)


def ell1_shearlet(
    shearlet: ShearletTransform,
    A: torch_radon.Radon,
    sinogram: torch.Tensor,
    p_0: float,
    p_1: float,
    Niter: int,
    ground_truth: Union[torch.Tensor, None],
    print_flag: bool = True,
) -> torch.Tensor:
    """
    Performs ℓ₁-regularized reconstruction using shearlet transforms and Radon projections.

    This algorithm aims to reconstruct an image `f` from a sinogram using an iterative
    optimization approach with ℓ₁-regularization in the shearlet domain.

    Args:
        shearlet (ShearletTransform): Shearlet transform object used for forward and backward transforms.
        A (torch_radon.Radon): Radon transform operator (forward and backward projections).
        sinogram (torch.Tensor): Input sinogram (measured projection data).
        p_0 (float): Weight for the Radon transform term in the optimization.
        p_1 (float): Weight for the shearlet regularization term.
        Niter (int): Number of iterations to run the optimization.
        ground_truth (Union[torch.Tensor, None]): Ground truth image for tracking reconstruction error
                                                 (used only if `print_flag` is True).
        print_flag (bool, optional): If True, prints the iteration number and reconstruction error (if `ground_truth` is provided).
                                     Defaults to True.

    Returns:
        torch.Tensor: The reconstructed image tensor.
    """

    bp = A.backward(sinogram)

    sc = shearlet.forward(bp)

    w = 3**shearlet.scales / 400
    w = shearlet.scales
    w = w.view(-1, 1, 1).cuda()

    u_1 = torch.zeros_like(sc)
    z_1 = torch.zeros_like(sc)
    u_2 = torch.zeros_like(bp)
    z_2 = torch.zeros_like(bp)

    f = torch.zeros_like(bp)
    for i in range(Niter):
        cg_y = p_0 * bp + p_1 * shearlet.backward(z_1 - u_1) + (z_2 - u_2)
        f = cg(
            lambda x: p_0 * A.backward(A.forward(x)) + (1 + p_1) * x,
            f.clone(),
            cg_y,
            max_iter=100,
        )
        sh_f = shearlet.forward(f)
        z_1 = shrink(sh_f + u_1, w * p_0 / p_1)
        z_2 = (f + u_2).clamp_min(0)
        u_1 = u_1 + sh_f - z_1
        u_2 = u_2 + f - z_2

        if print_flag:
            print(f"{i+1}/{Niter}, {torch.linalg.norm(f - ground_truth)}")

    return f
