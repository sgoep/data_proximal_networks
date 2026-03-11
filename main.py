#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.radon import RadonAdapter
from src.utils import mse_loss, set_seed, to_4d, build_models, save_example_outputs

from src.ellipse_dataloader import get_ellipse_dataloader
from src.lodopab_dataloader import get_lodopab_dataloader

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running, n = 0.0, 0
    for x_gt, x_init, y_delta in loader:
        x_gt = to_4d(x_gt).to(device)
        x_init = to_4d(x_init).to(device)
        y_delta = to_4d(y_delta).to(device)

        pred = model(x_init, y_delta)
        loss = mse_loss(pred, x_gt)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running += float(loss.item()) * x_gt.shape[0]
        n += x_gt.shape[0]
    return running / max(n, 1)


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    running, n = 0.0, 0
    for x_gt, x_init, y_delta in loader:
        x_gt = to_4d(x_gt).to(device)
        x_init = to_4d(x_init).to(device)
        y_delta = to_4d(y_delta).to(device)

        pred = model(x_init, y_delta)
        loss = mse_loss(pred, x_gt)

        running += float(loss.item()) * x_gt.shape[0]
        n += x_gt.shape[0]
    return running / max(n, 1)


def main(example):
    DATA_ROOT = f"{example}_out"
    OUT_DIR = Path(f"runs_{example}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    INIT_METHODS = ["tv", "lw"]
    MODELS_TO_TRAIN = ["resnet", "nsn", "dpnsn", "dpnsn_res"]

    # INIT_METHODS = ["lw", "tv"]
    # MODELS_TO_TRAIN = ["dpnsn"]

    # INIT_METHODS = ["tv", "lw"]
    # MODELS_TO_TRAIN = ["dpnsn_huber"]

    EPOCHS = 50
    BATCH_SIZE = 16
    LR = 1e-4
    NUM_WORKERS = 2

    summary_path = Path(DATA_ROOT) / "summary.json"
    with open(summary_path, "r") as f:
        summary = json.load(f)

    IMG_SIZE = int(summary["img_size"])
    NUM_ANGLES = int(summary["num_angles"])
    DET_COUNT = int(summary["det_count"])
    BETA = float(summary["mean_norm_y_minus_y_delta"])
    ANGLES = summary["angles"]
    PHI = summary["phi"]
    
    dx = summary["dx"]

    n_train = 4000
    n_test = 1000

    # -------------------------
    # Build radon geometry
    # -------------------------
    # angles = np.linspace(-np.pi/3, np.pi/3, NUM_ANGLES, endpoint=False).astype(np.float32)
    angles = np.asarray(ANGLES)
    # print(angles)
    phi = tuple(PHI)
    # print(phi)

    radon = RadonAdapter(
        resolution=IMG_SIZE,
        angles=angles,
        det_count=DET_COUNT,
        clip_to_circle=False,
        dx=dx,
        estimate_norm=False,
        phi=phi
    )

    print(f"Loaded summary from {summary_path}")
    print(f"IMG_SIZE={IMG_SIZE}, NUM_ANGLES={NUM_ANGLES}, DET_COUNT={DET_COUNT}, PHI={PHI}")
    print(f"BETA (mean y_diff_norms) = {BETA:.6e}")

    for init in INIT_METHODS:
        run_dir = OUT_DIR / f"init_{init}"
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "examples").mkdir(parents=True, exist_ok=True)

        if example == 'ellipses':
            train_loader = get_ellipse_dataloader(
                init_recon=init,
                batch_size=BATCH_SIZE,
                split="train",
                n_train=n_train,
                n_test=n_test,
                data_root=DATA_ROOT,
                shuffle=True,
                num_workers=NUM_WORKERS,
                device=None,
            )

            val_loader = get_ellipse_dataloader(
                init_recon=init,
                batch_size=BATCH_SIZE,
                split="test",
                n_train=n_train,
                n_test=n_test,
                data_root=DATA_ROOT,
                shuffle=False,
                num_workers=NUM_WORKERS,
                device=None,
            )
        else:
            train_loader = get_lodopab_dataloader(
                init_recon=init,
                batch_size=BATCH_SIZE,
                split="train",
                n_train=n_train,
                n_test=n_test,
                data_root=DATA_ROOT,
                shuffle=True,
                num_workers=NUM_WORKERS,
                device=None,
            )

            val_loader = get_lodopab_dataloader(
                init_recon=init,
                batch_size=BATCH_SIZE,
                split="test",
                n_train=n_train,
                n_test=n_test,
                data_root=DATA_ROOT,
                shuffle=False,
                num_workers=NUM_WORKERS,
                device=None,
            )

        models = build_models(MODELS_TO_TRAIN, radon=radon, beta=BETA)

        for name, model in models.items():
            model = model.to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)

            best_val = float("inf")
            ckpt_path = run_dir / "checkpoints" / f"{name}_best.pt"

            for epoch in range(1, EPOCHS + 1):
                tr = train_one_epoch(model, train_loader, optimizer, DEVICE)
                va = eval_one_epoch(model, val_loader, DEVICE)
                print(f"[init={init} | {name}] epoch {epoch:03d}/{EPOCHS} | train={tr:.6f} | val={va:.6f}")

                if va < best_val:
                    best_val = va
                    torch.save(
                        {
                            "init": init,
                            "model_name": name,
                            "state_dict": model.state_dict(),
                            "val_loss": best_val,
                            "epoch": epoch,
                        },
                        ckpt_path,
                    )

            print(f"[init={init} | {name}] best val={best_val:.6f} saved to {ckpt_path}")

            # save example recon output with best weights
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(ckpt["state_dict"])

            ex_path = run_dir / "examples" / f"{name}_example.png"
            save_example_outputs(
                model=model,
                loader=val_loader,
                device=DEVICE,
                out_path=ex_path,
                title=f"init={init} | model={name} | best_val={best_val:.6f}",
            )
            print(f"[init={init} | {name}] example saved to {ex_path}")


if __name__ == "__main__":
    main(example="ellipses")
    print("Finished.")

# sbatch -p a6000 -w mp-gpu4-a6000-2 --job-name=train -o logs/train.txt --time=30-00:00:00 --wrap="python -u main.py"
