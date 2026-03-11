from pathlib import Path
from typing import Optional, Tuple, Literal
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

InitType = Literal["tv", "fbp", "lw"]

class EllipsesGTInitDataset(Dataset):
    """
    Loads (ground_truth, init_reconstruction) pairs from ellipses_out/.
    init_reconstruction can be 'tv', 'fbp', or 'lw' (folder name).
    """

    def __init__(
        self,
        root: Path,
        init: InitType = "tv",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        check_files: bool = True,
    ):
        self.root = Path(root)
        self.gt_dir = self.root / "gt"
        self.init_dir = self.root / init
        self.sino_dir = self.root / "sino"
        
        self.files = sorted(f.name for f in self.gt_dir.glob("*.npy"))

        self.device = device
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fname = self.files[idx]

        x_gt = np.load(self.gt_dir / fname)
        x_init = np.load(self.init_dir / fname)
        y_delta = np.load(self.sino_dir / fname)

        # (H, W) -> (1, H, W)
        x_gt = torch.from_numpy(x_gt).unsqueeze(0).to(self.dtype)
        x_init = torch.from_numpy(x_init).unsqueeze(0).to(self.dtype)
        y_delta = torch.from_numpy(y_delta).unsqueeze(0).to(self.dtype)

        if self.device is not None:
            x_gt = x_gt.to(self.device, non_blocking=True)
            x_init = x_init.to(self.device, non_blocking=True)
            y_delta = y_delta.to(self.device, non_blocking=True)

        return x_gt, x_init, y_delta


def create_ellipses_dataloader(
    data_root: str = "ellipses_out",
    init: InitType = "tv",
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> DataLoader:
    dataset = EllipsesGTInitDataset(
        root=Path(data_root),
        init=init,
        device=device,
        dtype=dtype,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device is not None and device.type == "cuda"),
        drop_last=False,
    )


def get_ellipse_dataloader(
    init_recon: str,
    batch_size: int,
    data_root: str = "ellipses_out",
    split: str = "train",          # "train" or "test"
    n_train: int = 4000,
    n_test: int = 1000,
    shuffle: bool = True,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> DataLoader:
    init_recon = init_recon.lower()
    if init_recon not in ("tv", "fbp", "lw"):
        raise ValueError("init_recon must be one of: 'tv', 'fbp', 'lw'")

    # full dataset
    dataset = EllipsesGTInitDataset(
        root=Path(data_root),
        init=init_recon,
        device=device,
    )

    # deterministic split
    # rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    # rng.shuffle(indices)

    train_idx = indices[:n_train]
    test_idx = indices[n_train:n_train + n_test]

    if split == "train":
        subset = Subset(dataset, train_idx)
        do_shuffle = shuffle
    elif split == "test":
        subset = Subset(dataset, test_idx)
        do_shuffle = False
    else:
        raise ValueError("split must be 'train' or 'test'")

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=do_shuffle,
        num_workers=num_workers,
        pin_memory=(device is not None and device.type == "cuda"),
        drop_last=False,
    )