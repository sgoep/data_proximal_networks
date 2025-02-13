import torch
import torch_radon
from torch_radon.solvers import cg
from typing import Union

from .my_shearlet import ShearletTransform


def shrink(a, b):
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

    bp = A.backward(sinogram)

    sc = shearlet.forward(bp)

    w = 3**shearlet.scales / 400
    w = shearlet.scales
    # print(w)
    # w = torch.Tensor([0, 0.5, 1])
    w = w.view(-1, 1, 1).cuda()
    # print(w)
    # print(w.shape)

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
