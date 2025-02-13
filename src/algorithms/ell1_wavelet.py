import torch
import torch_radon
from pytorch_wavelets import DWTForward, DWTInverse
from torch_radon.solvers import cg
from typing import Union


def shrink(a, b):
    """
    Applies soft-thresholding (shrinkage) to enforce sparsity.

    Args:
        a (torch.Tensor): Input tensor.
        b (torch.Tensor): Threshold tensor.

    Returns:
        torch.Tensor: Shrunk tensor where values are reduced by `b` if they exceed `b`.
    """
    return (torch.abs(a) - b).clamp_min(0) * torch.sgn(a)


def shrink_(a, b):
    """
    Applies soft-thresholding to a tuple representing wavelet coefficients.

    Args:
        a (tuple): A tuple containing (approximation coefficients, list of detail coefficients).
        b (torch.Tensor): Threshold tensor.

    Returns:
        tuple: Thresholded wavelet coefficients (approximation, details).
    """
    a0 = shrink(a[0], b)
    an = []
    for ai in a[1]:
        an.append(shrink(ai, b))
    return (a0, an)


def copy_(x):
    """
    Creates a zero-initialized copy of a tuple representing wavelet coefficients.

    Args:
        x (tuple): A tuple containing (approximation coefficients, list of detail coefficients).

    Returns:
        tuple: A tuple of zero tensors with the same structure as `x`.
    """
    x0 = torch.zeros_like(x[0])
    xn = []
    for xi in x[1]:
        xn.append(torch.zeros_like(xi))
    return (x0, xn)


def my_diff(x, y):
    """
    Computes the element-wise difference between two sets of wavelet coefficients.

    Args:
        x (tuple): First set of wavelet coefficients.
        y (tuple): Second set of wavelet coefficients.

    Returns:
        tuple: Element-wise difference (approximation, details).
    """
    z0 = x[0] - y[0]
    zn = []
    for i in range(len(x[1])):
        zn.append(x[1][i] - y[1][i])
    return (z0, zn)


def my_add(x, y):
    """
    Computes the element-wise sum between two sets of wavelet coefficients.

    Args:
        x (tuple): First set of wavelet coefficients.
        y (tuple): Second set of wavelet coefficients.

    Returns:
        tuple: Element-wise sum (approximation, details).
    """
    z0 = x[0] + y[0]
    zn = []
    for i in range(len(x[1])):
        zn.append(x[1][i] + y[1][i])
    return (z0, zn)


def ell1_wavelet(
    wavelet: DWTForward,
    iwavelet: DWTInverse,
    A: torch_radon.Radon,
    sinogram: torch.Tensor,
    p_0: float,
    p_1: float,
    Niter: int,
    ground_truth: Union[torch.Tensor, None],
    print_flag: bool = True,
) -> torch.Tensor:
    """
    Performs ℓ₁-regularized reconstruction using wavelet transforms and Radon projections.

    This iterative optimization method reconstructs an image `f` from a sinogram by
    enforcing sparsity in the wavelet domain.

    Args:
        wavelet (DWTForward): Wavelet forward transform (discrete wavelet transform).
        iwavelet (DWTInverse): Wavelet inverse transform.
        A (torch_radon.Radon): Radon transform operator (forward and backward projections).
        sinogram (torch.Tensor): Input sinogram (measured projection data).
        p_0 (float): Weight for the Radon transform term in the optimization.
        p_1 (float): Weight for the wavelet regularization term.
        Niter (int): Number of iterations to run the optimization.
        ground_truth (Union[torch.Tensor, None]): Ground truth image for error tracking
                                                 (used only if `print_flag` is True).
        print_flag (bool, optional): If True, prints the iteration number and reconstruction error
                                     (if `ground_truth` is provided). Defaults to True.

    Returns:
        torch.Tensor: The reconstructed image tensor.
    """
    bp = A.backward(sinogram)

    # Apply the DWT forward transform
    sc = wavelet(bp.unsqueeze(0).unsqueeze(0))  # Add batch and channel dimensions

    # w = 3**len(sc[1]) / 400  # Adjust according to your scales
    # w = torch.Tensor([w]).view(-1, 1, 1).cuda()
    w = 1

    u_1 = copy_(sc)
    z_1 = copy_(sc)

    u_2 = torch.zeros_like(bp)
    z_2 = torch.zeros_like(bp)

    f = torch.zeros_like(bp)
    for i in range(Niter):
        cg_y = p_0 * bp + p_1 * iwavelet(my_diff(z_1, u_1)) + (z_2 - u_2)
        f = cg(
            lambda x: p_0 * A.backward(A.forward(x)) + (1 + p_1) * x,
            f.clone(),
            cg_y,
            max_iter=50,
        )

        sh_f = wavelet(f.unsqueeze(0).unsqueeze(0))  # Apply DWT
        z_1 = shrink_(
            sh_f + u_1, w * p_0 / p_1
        )  # Using only the approximation coefficients
        z_2 = (f + u_2).clamp_min(0)
        u_1 = my_add(u_1, my_diff(sh_f, z_1))
        u_2 = u_2 + f - z_2

        if print_flag:
            print(f"{i+1}/{Niter}, {torch.linalg.norm(f - ground_truth)}")

    return f
