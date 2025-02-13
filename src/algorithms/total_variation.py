from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_radon


def my_grad(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the gradient of a tensor `X`.

    Args:
        X (torch.Tensor): The input tensor.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Gradients in the x and y directions.
    """
    fx = torch.cat((X[1:, :], X[-1, :].unsqueeze(0)), dim=0) - X
    fy = torch.cat((X[:, 1:], X[:, -1].unsqueeze(1)), dim=1) - X
    return fx, fy


def my_div(Px: torch.Tensor, Py: torch.Tensor) -> torch.Tensor:
    """
    Compute the divergence of two tensors `Px` and `Py`.

    Args:
        Px (torch.Tensor): Gradient in the x direction.
        Py (torch.Tensor): Gradient in the y direction.

    Returns:
        torch.Tensor: The divergence of `Px` and `Py`.
    """
    fx = Px - torch.cat((Px[0, :].unsqueeze(0), Px[0:-1, :]), dim=0)
    fx[0, :] = Px[0, :]
    fx[-1, :] = -Px[-2, :]

    fy = Py - torch.cat((Py[:, 0].unsqueeze(1), Py[:, 0:-1]), dim=1)
    fy[:, 0] = Py[:, 0]
    fy[:, -1] = -Py[:, -2]

    return fx + fy
    # # Divergence along x (horizontal)
    # div_x = Px[:, 1:] - Px[:, :-1]
    # div_x = torch.cat((torch.zeros_like(div_x[:, :1]), div_x), dim=-1)  # Pad to keep same shape

    # # Divergence along y (vertical)
    # div_y = Py[1:, :] - Py[:-1, :]
    # div_y = torch.cat((torch.zeros_like(div_y[:1, :]), div_y), dim=-2)  # Pad to keep same shape

    # return div_x + div_y


def tv(
    x0: torch.Tensor,
    A: torch_radon.Radon,
    g: torch.Tensor,
    alpha: float,
    tau: float,
    sigma: float,
    theta: float,
    L: float,
    Niter: int,
    ground_truth: torch.Tensor = None,
    print_flag: bool = True,
) -> torch.Tensor:
    """
    Perform Total Variation (TV) regularization.

    Args:
        x0 (torch.Tensor): Initial guess for the denoising process.
        A (torch_radon.Radon): Radon transform object.
        g (torch.Tensor): Sinogram (observed data).
        alpha (float): Regularization parameter.
        L (float): Lipschitz constant.
        Niter (int): Number of iterations.
        ground_truth (torch.Tensor, optional): Ground truth image for error
        calculation.
        print_flag (bool, optional): Flag to print error every 100 iterations.

    Returns:
        torch.Tensor: The denoised image.
    """

    # tau = 1/L
    # tau = 0.001
    # sigma = 1/L
    # sigma = 0.001
    # theta = 1
    grad_scale = 1e2

    p = torch.zeros_like(g).to("cuda")
    qx = x0
    qy = x0
    u = x0
    uiter = x0
    ubar = x0

    alpha = torch.Tensor([alpha]).to("cuda")
    zero_t = torch.Tensor([0]).to("cuda")

    error = torch.zeros(Niter)
    for k in range(Niter):

        p = (p + sigma * (A.forward(ubar) - g)) / (1 + sigma)

        ubarx, ubary = my_grad(ubar)
        if alpha > 0:

            qx = (alpha * (qx + grad_scale * sigma * ubarx)) / torch.maximum(
                alpha, torch.abs(qx + grad_scale * sigma * ubarx)
            )
            # print(qx)
            qy = (alpha * (qy + grad_scale * sigma * ubary)) / torch.maximum(
                alpha, torch.abs(qy + grad_scale * sigma * ubary)
            )

            # qx = alpha * (qx + grad_scale * sigma * ubarx)/torch.maximum(torch.sqrt(torch.sum(qx**2, 0))/alpha, ones)
            # qy = alpha * (qy + grad_scale * sigma * ubary)/torch.maximum(torch.sqrt(torch.sum(qy**2, 0))/alpha, ones)

            uiter = torch.maximum(
                zero_t, u - tau * (A.backward(p) - grad_scale * my_div(qx, qy))
            )
            # uiter = u - tau * (A.backward(p) + grad_scale * my_div(qx, qy))
        else:
            uiter = torch.maximum(zero_t, u - tau * A.backward(p))

        ubar = uiter + theta * (uiter - u)
        u = ubar

        if ground_truth is not None:
            difference = torch.abs(ubar - ground_truth)
            squared_difference = difference**2
            ground_truth_squared = torch.abs(ground_truth) ** 2
            denominator = torch.sum(ground_truth_squared)
            result = squared_difference / denominator
            error[k] = torch.sum(result)
        if print_flag and np.mod(k + 1, 100) == 0:
            print(
                f"TV Iteration: {str(k+1)} / {str(Niter)}, "
                f"Error: {str(error[k].item())}"
            )

    return ubar


if __name__ == "__main__":
    print("Running TV example reconstruction.")
    from src.utils.load_single_image import load_single_image

    print("Load image.")
    device = "cuda"
    phantom = load_single_image()

    x = torch.Tensor(phantom).to(device)
    Nal = 80
    angles = np.linspace(-np.pi / 3, np.pi / 3, Nal, endpoint=False)
    NUM_ANGLES = len(angles)

    print("Create Radon operator and data.")
    radon = torch_radon.Radon(
        128, angles, det_count=128, det_spacing=1, clip_to_circle=True
    )

    sinogram = radon.forward(x)
    noise = torch.randn(*sinogram.shape).to(device)
    sinogram += 0.03 * torch.max(torch.abs(sinogram)) * noise

    print("Run TV regularization.")
    x0 = torch.zeros([128, 128]).to(device)
    L = 400
    Niter = 500
    alpha = 0.04
    rec = tv(x0, radon, sinogram, alpha, L, Niter, ground_truth=x, print_flag=True)

    print("Plotting.")
    plt.figure()
    plt.imshow(rec.cpu().numpy())
    plt.colorbar()
    plt.savefig("tv_example.png")
    print("Finished.")
