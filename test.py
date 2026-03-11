#!/usr/bin/env python3
from pathlib import Path
import json
import math
import re
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

from src.radon import RadonAdapter
from src.utils import set_seed, to_4d, build_models, rel_l2_np, ssim, psnr
from src.ellipse_dataloader import get_ellipse_dataloader
from src.lodopab_dataloader import get_lodopab_dataloader






def find_checkpoints(runs_dir: Path):
    ckpts = []
    for init_dir in sorted(runs_dir.glob("init_*")):
        init = init_dir.name.replace("init_", "", 1)
        ckpt_dir = init_dir / "checkpoints"
        if not ckpt_dir.exists():
            continue
        for p in sorted(ckpt_dir.glob("*_best.pt")):
            name = p.stem.replace("_best", "")
            ckpts.append((init, name, p))
    return ckpts


@torch.no_grad()
def eval_model_on_loader(model, loader, device):
    rows = []
    idx = 0
    for x_gt, x_init, y_delta in loader:
        x_gt = to_4d(x_gt).to(device)
        x_init = to_4d(x_init).to(device)
        y_delta = y_delta.to(device)
        while y_delta.ndim > 4:
            y_delta = y_delta.squeeze(2)
        if y_delta.ndim == 3:
            y_delta = y_delta.unsqueeze(1)

        pred = model(x_init, y_delta)

        B = x_gt.shape[0]
        for b in range(B):
            gt = x_gt[b, 0].detach().cpu().numpy()
            pr = pred[b, 0].detach().cpu().numpy()

            rows.append(
                {
                    "index": idx,
                    "rel_l2": rel_l2_np(pr, gt),
                    "psnr": psnr(pr, gt),
                    "ssim": ssim(pr, gt),
                }
            )
            idx += 1
    return rows


def main(example: str = "ellipses"):
    DATA_ROOT = f"{example}_out"
    RUNS_DIR = Path(f"runs_{example}")
    OUT_CSV = RUNS_DIR / f"test_metrics_{example}.csv"
    OUT_SUMMARY_CSV = RUNS_DIR / f"test_metrics_{example}_summary.csv"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    summary_path = Path(DATA_ROOT) / "summary.json"
    with open(summary_path, "r") as f:
        summary = json.load(f)

    IMG_SIZE = int(summary["img_size"])
    DET_COUNT = int(summary["det_count"])
    BETA = float(summary["mean_norm_y_minus_y_delta"])
    angles = np.asarray(summary["angles"])
    phi = tuple(summary["phi"])
    dx = float(summary.get("dx", 1.0))

    radon = RadonAdapter(
        resolution=IMG_SIZE,
        angles=angles,
        det_count=DET_COUNT,
        clip_to_circle=False,
        dx=dx,
        estimate_norm=False,
        phi=phi,
        device=DEVICE,
        dtype=torch.float32,
    )

    ckpts = find_checkpoints(RUNS_DIR)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under {RUNS_DIR}/init_*/checkpoints/*_best.pt")

    all_rows = []

    for init, model_name, ckpt_path in ckpts:
        if example == "ellipses":
            test_loader = get_ellipse_dataloader(
                init_recon=init,
                batch_size=16,
                split="test",
                n_train=summary.get("n_train", 4000),
                n_test=summary.get("n_test", 1000),
                data_root=DATA_ROOT,
                shuffle=False,
                num_workers=2,
                device=None,
            )
        else:
            test_loader = get_lodopab_dataloader(
                init_recon=init,
                batch_size=16,
                split="test",
                n_train=summary.get("n_train", 4000),
                n_test=summary.get("n_test", 1000),
                data_root=DATA_ROOT,
                shuffle=False,
                num_workers=2,
                device=None,
            )

        models = build_models([model_name], radon=radon, beta=BETA)
        model = models[model_name].to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        rows = eval_model_on_loader(model, test_loader, DEVICE)
        for r in rows:
            r["example"] = example
            r["init"] = init
            r["model"] = model_name
            r["checkpoint"] = str(ckpt_path)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)

    summary_df = (
        df.groupby(["example", "init", "model"], as_index=False)
        .agg(
            n=("index", "count"),
            rel_l2_mean=("rel_l2", "mean"),
            rel_l2_std=("rel_l2", "std"),
            psnr_mean=("psnr", "mean"),
            psnr_std=("psnr", "std"),
            ssim_mean=("ssim", "mean"),
            ssim_std=("ssim", "std"),
        )
    )
    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)

    print("saved:", OUT_CSV)
    print("saved:", OUT_SUMMARY_CSV)

if __name__ == "__main__":
    main(example="ellipses")

# sbatch -p a6000 -w mp-gpu4-a6000-2 --job-name=test -o logs/test.txt --time=30-00:00:00 --wrap="python -u test.py"
