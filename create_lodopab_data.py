# %%
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from dival.datasets import LoDoPaBDataset

from src.radon import RadonAdapter
from src.total_variation import tv_cp
from src.landweber import landweber
from src.utils import ensure_dir, set_seed, rel_l2, to_4d
import matplotlib.pyplot as plt

def main():
    OUT_DIR = Path("lodopab_out")
    PART = "train"
    N_SAMPLES = 5000
    TV_SUBSET = 100

    ADD_NOISE = False
    NOISE_sigma_REL = 0.0

    TV_ALPHA_GRID = np.logspace(np.log10(0.08), np.log10(0.9), 15)#np.logspace(-6, -2, 10)
    TV_ITERS_SELECT = 500
    TV_ITERS_FINAL = 500

    LW_ITERS = 200
    LW_OMEGA_FACTOR = 1.0

    THETA = 1.0
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(0)

    # ----------------------------
    # output structure
    # ----------------------------
    ensure_dir(OUT_DIR)
    ensure_dir(OUT_DIR / "gt")
    ensure_dir(OUT_DIR / "fbp")
    ensure_dir(OUT_DIR / "tv")
    ensure_dir(OUT_DIR / "lw")
    ensure_dir(OUT_DIR / "sino")

    # ----------------------------
    # dataset
    # ----------------------------
    dataset = LoDoPaBDataset(impl="astra_cpu")  # make sure dival config points to your LoDoPaB data

    # take one sample to infer geometry
    sino0_np, gt0_np = dataset.get_sample(0, part=PART)  # (observation, ground_truth)
    n_angles, det_count = sino0_np.shape
    resolution = gt0_np.shape[-1]

    angles = np.linspace(-np.pi/2, np.pi/2, n_angles)
    phi = (-np.pi/3, np.pi/3)

    dx = 2 * 0.13 / resolution

    radon = RadonAdapter(
        resolution=resolution,
        angles=angles,
        det_count=det_count,
        clip_to_circle=False,
        dx=dx,
        estimate_norm=True,
        norm_iters=20,
        device=torch.device(DEVICE),
        dtype=torch.float32,
        phi=phi
    )
    
    L = 1e3#radon.norm_A2
    tau, sigma = 1 / L, 1 / L
    omega = (LW_OMEGA_FACTOR / radon.norm_A2)

    print(f"Generating {N_SAMPLES} samples from LoDoPaB part='{PART}' ...")

    samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
    y_diff_norms: List[float] = []

    for i in range(N_SAMPLES):
        sino_np, gt_np = dataset.get_sample(i, part=PART)

        gt = torch.from_numpy(gt_np.data).to(DEVICE)                 # (H,W)
        y = torch.from_numpy(sino_np.data).to(DEVICE)                # (angles,det)
        y = y.unsqueeze(0).unsqueeze(0)
        # plt.figure()
        # plt.imshow(y.squeeze().detach().cpu().numpy())
        # plt.savefig('y1.png')
        y = radon.proj_ran(y)
        # plt.figure()
        # plt.imshow(y.squeeze().detach().cpu().numpy())
        # plt.savefig('y2.png')
        
        y_gt = radon.proj_ran(radon.forward(gt))
        y_delta = y
        y_diff_norms.append(float(torch.linalg.norm((y_gt - y_delta).reshape(-1))))

        x_fbp = radon.fbp(y_delta, filter_name="ram-lak").squeeze()  # (H,W)

        np.save(OUT_DIR / "gt" / f"{i:05d}.npy", gt.detach().cpu().numpy())
        np.save(OUT_DIR / "sino" / f"{i:05d}.npy", y_delta.squeeze().detach().cpu().numpy())
        np.save(OUT_DIR / "fbp" / f"{i:05d}.npy", x_fbp.detach().cpu().numpy())

        samples.append((gt, y_delta))

    y_diff_norms = np.array(y_diff_norms, dtype=np.float32)
    np.save(OUT_DIR / "y_diff_norms.npy", y_diff_norms)

    # ----------------------------
    # select TV alpha on subset
    # ----------------------------
    subset = samples[:TV_SUBSET]

    print("Selecting TV alpha...")
    alpha_errors = {}

    for alpha in TV_ALPHA_GRID:
        errs = []
        for x_gt, y_delta in subset:
            x0 = radon.fbp(y_delta, filter_name="ram-lak")
            x_tv = tv_cp(
                x0=x0,
                A=radon.forward_la,
                AT=radon.backward_la,
                g=y_delta,
                alpha=float(alpha),
                tau=tau,
                sigma=sigma,
                theta=THETA,
                Niter=TV_ITERS_SELECT,
                print_flag=False,
            ).squeeze()
            errs.append(rel_l2(x_tv, x_gt))
        alpha_errors[float(alpha)] = float(np.mean(errs))
        print(f"alpha={alpha:.3e}: mean rel L2 = {alpha_errors[float(alpha)]:.4e}")

    best_alpha = min(alpha_errors, key=alpha_errors.get)
    print(f"Best alpha: {best_alpha:.6g}")

    # ----------------------------
    # final TV + Landweber
    # ----------------------------
    print("Running final TV and Landweber reconstructions...")
    for i, (x_gt, y_delta) in enumerate(samples):
        x0 = radon.fbp(y_delta, filter_name="ram-lak")

        x_tv = tv_cp(
            x0=x0,
            A=radon.forward_la,
            AT=radon.backward_la,
            g=y_delta,
            alpha=float(best_alpha),
            tau=tau,
            sigma=sigma,
            theta=THETA,
            Niter=TV_ITERS_FINAL,
            print_flag=False,
        ).squeeze()

        x_lw = landweber(
            A=radon.forward_la,
            AT=radon.backward_la,
            g=y_delta,
            x0=x0,
            omega=float(omega),
            n_iter=LW_ITERS,
        ).squeeze()

        np.save(OUT_DIR / "tv" / f"{i:05d}.npy", x_tv.detach().cpu().numpy())
        np.save(OUT_DIR / "lw" / f"{i:05d}.npy", x_lw.detach().cpu().numpy())

    # ----------------------------
    # summary
    # ----------------------------
    summary = {
        "dataset": "lodopab",
        "part": PART,
        "n_samples": N_SAMPLES,
        "img_size": int(resolution),
        "num_angles": int(n_angles),
        "angles": angles.tolist(),
        "phi": list(phi),
        "det_count": int(det_count),
        "dx": float(dx),
        "device": DEVICE,
        "add_noise": bool(ADD_NOISE),
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

    print("Done. Data saved to:", (OUT_DIR).resolve())


if __name__ == "__main__":
    main()

# sbatch -p a6000 -w mp-gpu4-a6000-3 --job-name=lodopab_data -o logs/lodopab_data.txt --time=30-00:00:00 --wrap="python -u create_lodopab_data.py"

# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

idx = 0
base = Path("lodopab_out/")

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
