import torch
import numpy as np
from pathlib import Path
from src.unet import UNet
from src.wrappers import RESNET, NSN, DPNSN, DPNSN_RES
from typing import List, Union, Dict
from src.radon import RadonAdapter
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

@torch.no_grad()
def save_example_outputs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_path: Path,
    title: str,
):
    model.eval()
    x_gt, x_init, y_delta = next(iter(loader))
    x_gt = to_4d(x_gt).to(device)
    x_init = to_4d(x_init).to(device)
    y_delta = to_4d(y_delta).to(device)

    pred = model(x_init, y_delta)

    gt_np = x_gt[0, 0].detach().cpu().numpy()
    init_np = x_init[0, 0].detach().cpu().numpy()
    pred_np = pred[0, 0].detach().cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    im0 = axes[0].imshow(gt_np, cmap="gray")
    axes[0].set_title("GT")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(init_np, cmap="gray")
    axes[1].set_title("Init")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(pred_np, cmap="gray")
    axes[2].set_title("Model Output")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

def rel_l2(x: torch.Tensor, x_gt: torch.Tensor, eps: float = 1e-12) -> float:
    num = torch.linalg.norm((x - x_gt).reshape(-1))
    den = torch.linalg.norm(x_gt.reshape(-1)).clamp_min(eps)
    return float((num / den).item())

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def create_simple_phantom(size: int, device="cpu"):
    """
    Simple phantom:
    - filled disk
    - rectangle
    """
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, size, device=device),
        torch.linspace(-1, 1, size, device=device),
        indexing="ij",
    )

    img = torch.zeros((size, size), device=device)

    # disk
    img[(xx**2 + yy**2) < 0.5**2] = 1.0

    # rectangle
    img[(xx > -0.7) & (xx < -0.3) & (yy > -0.2) & (yy < 0.4)] = 0.7
    
    return img

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_4d(x: torch.Tensor) -> torch.Tensor:
    """Ensure shape is (B, 1, H, W)."""
    if x.ndim == 2:
        return x.unsqueeze(0).unsqueeze(0)
    if x.ndim == 3:
        return x.unsqueeze(1)
    return x


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)

def save_image_with_colorbar(img2d: np.ndarray, out_png: Path, title: str) -> None:
    plt.figure(figsize=(5, 4))
    im = plt.imshow(img2d, cmap="gray")
    plt.colorbar(im)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    
def build_models(
    which: List[str],
    radon: RadonAdapter,
    beta: Union[float, None] = None,
) -> Dict[str, nn.Module]:
    models: Dict[str, nn.Module] = {}
    for name in which:
        name = name.lower()
        if name == "resnet":
            models[name] = RESNET(unet=UNet(in_channels=1, out_channels=1))
        elif name == "nsn":
            models[name] = NSN(unet=UNet(in_channels=1, out_channels=1), radon=radon)
        elif name == "dpnsn":
            models[name] = DPNSN(unet=UNet(in_channels=1, out_channels=1), radon=radon, beta=beta)
        elif name == "dpnsn_res":
            models[name] = DPNSN_RES(unet=UNet(in_channels=1, out_channels=1), radon=radon, beta=beta)
        else:
            raise ValueError(f"Unknown model '{name}'. Use one of: resnet, nsn, dpdnsn")
    return models
