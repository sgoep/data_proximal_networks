# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.radon import RadonAdapter
from typing import Optional
import torch


def grad_forward(u: torch.Tensor):
    """
    Forward differences with Neumann boundary.
    Works for (..., H, W).
    Returns (gx, gy) same shape as u.
    """
    gx = torch.zeros_like(u)
    gy = torch.zeros_like(u)

    # d/dx (vertical, H axis)
    if u.shape[-2] > 1:
        gx[..., :-1, :] = u[..., 1:, :] - u[..., :-1, :]
        gx[..., -1, :] = 0.0

    # d/dy (horizontal, W axis)
    if u.shape[-1] > 1:
        gy[..., :, :-1] = u[..., :, 1:] - u[..., :, :-1]
        gy[..., :, -1] = 0.0

    return gx, gy


def div_backward(px: torch.Tensor, py: torch.Tensor):
    """
    Divergence (negative adjoint of grad_forward) with matching boundaries.
    Works for (..., H, W).
    """
    div = torch.zeros_like(px)

    H = px.shape[-2]
    W = px.shape[-1]

    # x-part
    if H == 1:
        div_x = torch.zeros_like(px)
    else:
        div_x = torch.zeros_like(px)
        div_x[..., 0, :] = px[..., 0, :]
        div_x[..., 1:-1, :] = px[..., 1:-1, :] - px[..., 0:-2, :]
        div_x[..., -1, :] = -px[..., -2, :]

    # y-part
    if W == 1:
        div_y = torch.zeros_like(py)
    else:
        div_y = torch.zeros_like(py)
        div_y[..., :, 0] = py[..., :, 0]
        div_y[..., :, 1:-1] = py[..., :, 1:-1] - py[..., :, 0:-2]
        div_y[..., :, -1] = -py[..., :, -2]

    div = div_x + div_y
    return div

# --------- TV solver (Chambolle–Pock style) ---------

@torch.no_grad()
def tv_cp(
    x0: torch.Tensor,
    A,                    # RadonAdapter-like: forward/backward
    AT,
    g: torch.Tensor,
    alpha: float,
    tau: float,
    sigma: float,
    theta: float,
    Niter: int,
    ground_truth: Optional[torch.Tensor] = None,
    print_flag: bool = True,
    grad_scale: float = 1e2,
) -> torch.Tensor:

    device = x0.device
    dtype = x0.dtype

    # primal
    u = x0.clone()
    ubar = u.clone()

    # dual variables
    p = torch.zeros_like(g)                 # sinogram dual
    qx = torch.zeros_like(u)                # TV dual (x)
    qy = torch.zeros_like(u)                # TV dual (y)

    alpha_t = torch.tensor(alpha, device=device, dtype=dtype)
    inv_1psigma = 1.0 / (1.0 + sigma)

    # optional error tracking
    if ground_truth is not None:
        gt_denom = torch.sum(ground_truth.abs().pow(2)).clamp_min(1e-12)

    for k in range(Niter):
        
        Au = A(ubar)
        p.add_(sigma * (Au - g)).mul_(inv_1psigma)

        # --- dual update for TV term ---
        if alpha > 0.0:
            ux, uy = grad_forward(ubar)

            qx.add_(grad_scale * sigma * ux)
            qy.add_(grad_scale * sigma * uy)

            qx.copy_(alpha_t * qx / torch.maximum(alpha_t, qx.abs()))
            qy.copy_(alpha_t * qy / torch.maximum(alpha_t, qy.abs()))

            div_q = div_backward(qx, qy)

            # primal update: u^{k+1} = max(0, u^k - tau*(A^T p - grad_scale*div(q)))
            u_next = u - tau * (AT(p) - grad_scale * div_q)
            u_next.clamp_min_(0.0)

        else:
            u_next = u - tau * AT(p)
            u_next.clamp_min_(0.0)

        # extrapolation
        ubar = u_next + theta * (u_next - u)
        u = u_next

        if print_flag and (k + 1) % 100 == 0:
            if ground_truth is not None:
                err = torch.sum((ubar - ground_truth).abs().pow(2)) / gt_denom
                print(f"TV Iteration: {k+1}/{Niter}, Error: {err.item():.6g}")
            else:
                print(f"TV Iteration: {k+1}/{Niter}")

    return ubar

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    from src.utils import create_simple_phantom
    from src.radon import RadonAdapter  # <- your adapter (the one with norm estimation)
    # from src.utils.radon_operator import filter_sinogram  # only needed if your tv_cp uses it internally

    print("Running TV example reconstruction.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Load image.")
    x = create_simple_phantom(128, device=device).to(device)  # (H,W) or (1,1,H,W) depending on your helper
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)  # -> (1,1,H,W)

    Nal = 80
    angles = np.linspace(-np.pi / 3, np.pi / 3, Nal, endpoint=False)

    print("Create RadonAdapter and data.")
    radon = RadonAdapter(
        resolution=128,
        angles=angles,
        det_count=128,
        clip_to_circle=False,
        dx=1.0,
        estimate_norm=True,
        norm_iters=20,
        device=device,
        dtype=torch.float32,
    )

    sinogram = radon.forward(x)

    noise = torch.randn_like(sinogram)
    sinogram = sinogram + 0.03 * sinogram.abs().max() * noise

    print("Run TV regularization.")

    x0 = torch.zeros_like(x)

    alpha = 0.04

    L = radon.norm_A2
    # print(L)
    # L = 400
    tau, sigma = 1/L, 1/L
    theta = 1.0
    Niter = 5000

    rec = tv_cp(
        x0=x0,
        A=radon,
        g=sinogram,
        alpha=alpha,
        tau=float(tau),
        sigma=float(sigma),
        theta=theta,
        Niter=Niter,
        ground_truth=x,
        print_flag=True,
    )

    # rec should be (B,C,H,W); plot first image
    rec_img = rec[0, 0].detach().cpu().numpy()

    print("Plotting.")
    plt.figure(figsize=(5, 4))
    plt.imshow(rec_img, cmap="gray")
    plt.colorbar()
    plt.title("TV Reconstruction (RadonAdapter)")
    plt.tight_layout()
    plt.savefig("tv_example.png", dpi=150)
    plt.close()

    print("Finished.")
    
# sbatch -p a6000 -w mp-gpu4-a6000-2 --job-name=tv -o logs/tv.txt --time=30-00:00:00 --wrap="python -u -m src.total_variation"
