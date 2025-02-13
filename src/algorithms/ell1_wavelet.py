import torch
import torch_radon
from pytorch_wavelets import DWTForward, DWTInverse
from torch_radon.solvers import cg
from typing import Union


def shrink(a, b):
    return (torch.abs(a) - b).clamp_min(0) * torch.sgn(a)


def shrink_(a, b):
    a0 = shrink(a[0], b)
    an = []
    for ai in a[1]:
        an.append(shrink(ai, b))
    return (a0, an)


def copy_(x):
    x0 = torch.zeros_like(x[0])
    xn = []
    for xi in x[1]:
        xn.append(torch.zeros_like(xi))
    return (x0, xn)


def my_diff(x, y):
    z0 = x[0] - y[0]
    zn = []
    for i in range(len(x[1])):
        zn.append(x[1][i] - y[1][i])
    return (z0, zn)


def my_add(x, y):
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
