#!/usr/bin/env python3
# prepare_ellipses_fbp_tv.py

import os
import json
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.radon import RadonAdapter
# DIVAL Ellipses
from dival.datasets import EllipsesDataset

from src.utils import ensure_dir, set_seed, rel_l2, save_image_with_colorbar, to_4d
from src.total_variation import tv_cp
from src.landweber import landweber


def main():
    OUT_DIR = Path("ellipses_out")
    N_SAMPLES = 5000
    TV_SUBSET = 100

    IMG_SIZE = 128
    NUM_ANGLES = 180
    DET_COUNT = int(np.sqrt(2)*IMG_SIZE) + 1

    NOISE_sigma_REL = 0.02

    TV_ALPHA_GRID = np.logspace(np.log10(0.001), np.log10(1.0), 20)#np.logspace(-5, -1, 10)#[0.002, 0.005, 0.01, 0.02, 0.04]
    TV_ITERS_SELECT = 200
    TV_ITERS_FINAL = 200
    
    LW_ITERS = 200
    LW_OMEGA_FACTOR = 1.0

    THETA = 1.0

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(0)

    # output structure
    ensure_dir(OUT_DIR)
    ensure_dir(OUT_DIR / "gt")
    ensure_dir(OUT_DIR / "fbp")
    ensure_dir(OUT_DIR / "tv")
    ensure_dir(OUT_DIR / "lw")
    ensure_dir(OUT_DIR / "sino")

    # dataset
    dataset = EllipsesDataset(image_size=IMG_SIZE)
    gen = dataset.generator("train")

    # radon
    dx = 1.0
    # angles = np.linspace(-np.pi/2, np.pi/2, NUM_ANGLES, endpoint=False).astype(np.float32)
    angles = np.arange(-90, 90) * np.pi/180
    phi = (-np.pi/3, np.pi/3)
    radon = RadonAdapter(
        resolution=IMG_SIZE,
        angles=angles,
        det_count=DET_COUNT,
        clip_to_circle=False,
        dx=dx,
        phi=phi
    )
    
    # L = radon.norm_A2
    L = 400
    tau, sigma = 1/L, 1/L
    
    omega = (LW_OMEGA_FACTOR / radon.norm_A2)

    y_diff_norms: List[float] = []
    subset = []

    print("Generating data...")
    samples: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for i in range(N_SAMPLES):
        x_gt = torch.from_numpy(next(gen).data).to(DEVICE)

        y = radon.forward_la(to_4d(x_gt))
        noise = radon.proj_ran(torch.randn_like(y))
        y_delta = y + NOISE_sigma_REL * y.abs().max() * noise

        y_diff_norms.append(float(torch.linalg.norm((y - y_delta).reshape(-1))))

        x_fbp = radon.fbp_la(y_delta).squeeze()

        np.save(OUT_DIR / "gt" / f"{i:05d}.npy", x_gt.detach().cpu().numpy())
        np.save(OUT_DIR / "fbp" / f"{i:05d}.npy", x_fbp.detach().cpu().numpy())
        np.save(OUT_DIR / "sino" / f"{i:05d}.npy", y_delta.squeeze().detach().cpu().numpy())

        samples.append((x_gt, y_delta))

    y_diff_norms = np.array(y_diff_norms)
    np.save(OUT_DIR / "y_diff_norms.npy", y_diff_norms)

    subset = samples[:TV_SUBSET]

    print("Selecting TV alpha...")
    alpha_errors = {}

    for alpha in TV_ALPHA_GRID:
        errs = []
        for x_gt, y_delta in subset:
            x0 = radon.fbp(y_delta)
            x_tv = tv_cp(
                x0=x0,
                A=radon.forward_la,
                AT=radon.backward_la,
                g=y_delta,
                alpha=alpha,
                tau=tau,
                sigma=sigma,
                theta=THETA,
                Niter=TV_ITERS_SELECT,
                print_flag=False,
                # grad_scale=0
            ).squeeze()
            errs.append(rel_l2(x_tv, x_gt))
        alpha_errors[alpha] = float(np.mean(errs))
        print(f"alpha={alpha}: mean rel L2 = {alpha_errors[alpha]:.4e}")

    best_alpha = min(alpha_errors, key=alpha_errors.get)
    print(f"Best alpha: {best_alpha}")

    print("Running final TV reconstructions...")
    for i, (x_gt, y_delta) in enumerate(samples):
        x0 = radon.fbp(y_delta)
        x_tv = tv_cp(
            x0=x0,
            A=radon.forward_la,
            AT=radon.backward_la,
            g=y_delta,
            alpha=best_alpha,
            tau=tau,
            sigma=sigma,
            theta=THETA,
            Niter=TV_ITERS_FINAL,
            print_flag=False,
            # grad_scale=0
        ).squeeze()
        
        x_lw = landweber(
            A=radon.forward_la,
            AT=radon.backward_la,
            g=y_delta,
            x0=x0,
            omega=omega,
            n_iter=LW_ITERS,
        ).squeeze()

        np.save(OUT_DIR / "tv" / f"{i:05d}.npy", x_tv.detach().cpu().numpy())
        np.save(OUT_DIR / "lw" / f"{i:05d}.npy", x_lw.detach().cpu().numpy())

    # summary = {
    #     "n_samples": N_SAMPLES,
    #     "noise_sigma_rel": NOISE_sigma_REL,
    #     "mean_norm_y_minus_y_delta": float(y_diff_norms.mean()),
    #     "tv_alpha_grid": TV_ALPHA_GRID.tolist(),
    #     "tv_alpha_errors_subset": alpha_errors,
    #     "tv_best_alpha": best_alpha,
    #     "lw_iters": LW_ITERS,
    #     "lw_omega": omega,
    #     "lw_omega_factor": LW_OMEGA_FACTOR,
    #     "img_size": IMG_SIZE,
    #     "num_angles": NUM_ANGLES,
    #     "angles": angles.tolist(),
    #     "phi": list(phi),
    #     "det_count": DET_COUNT,
    #     "device": DEVICE,
    # }
    summary = {
        "dataset": "ellipse",
        "part": None,
        "n_samples": N_SAMPLES,
        "img_size": int(IMG_SIZE),
        "num_angles": int(NUM_ANGLES),
        "det_count": int(DET_COUNT),
        "angles": angles.tolist(),
        "dx": float(dx),
        "phi": list(phi),
        "device": DEVICE,
        "add_noise": None,
        "noise_sigma_rel": float(NOISE_sigma_REL),
        "mean_norm_y_minus_y_delta": float(y_diff_norms.mean()),
        "tv_alpha_grid": [float(a) for a in TV_ALPHA_GRID.tolist()],
        "tv_alpha_errors_subset": alpha_errors,
        "tv_best_alpha": float(best_alpha),
        "tv_iters_select": int(TV_ITERS_SELECT),
        "tv_iters_final": int(TV_ITERS_FINAL),
        "lw_iters": int(LW_ITERS),
        "lw_omega": float(omega),
        "lw_omega_factor": float(LW_OMEGA_FACTOR),
        "operator_norm_A2": float(L),
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Done. Data saved to:", OUT_DIR.resolve())



if __name__ == "__main__":
    main()


# sbatch -p a6000 -w mp-gpu4-a6000-2 --job-name=ellipse_data -o logs/ellipse_data.txt --time=30-00:00:00 --wrap="python -u create_ellipse_data.py"

# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

idx = 0
base = Path("ellipses_out")

x_gt = np.load(base / "gt" / f"{idx:05d}.npy")
x_fbp = np.load(base / "fbp" / f"{idx:05d}.npy")
x_tv = np.load(base / "tv" / f"{idx:05d}.npy")
x_lw = np.load(base / "lw" / f"{idx:05d}.npy")

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

im0 = axes[0].imshow(x_gt, cmap="gray"); axes[0].set_title("GT"); axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(x_fbp, cmap="gray"); axes[1].set_title("FBP"); axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(x_tv, cmap="gray"); axes[2].set_title("TV"); axes[2].axis("off")
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

im3 = axes[3].imshow(x_lw, cmap="gray"); axes[3].set_title("Landweber"); axes[3].axis("off")
plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
