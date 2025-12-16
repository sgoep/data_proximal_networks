# %%
from pathlib import Path
from typing import Optional, Tuple, Literal
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

InitType = Literal["tv", "fbp", "lw"]
PartType = Literal["train", "validation", "test"]


from pathlib import Path
from typing import Optional, Tuple, Literal
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

InitType = Literal["tv", "fbp", "lw"]


class LoDoPaBGTInitDataset(Dataset):
    """
    Loads (ground_truth, init_reconstruction) pairs from lodopab_out/.
    Expects directories: gt/, fbp/, tv/, lw/
    """

    def __init__(
        self,
        root: Path,
        init: str = "tv",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
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

        x_gt = torch.from_numpy(x_gt).unsqueeze(0).to(self.dtype)      # (1,H,W)
        x_init = torch.from_numpy(x_init).unsqueeze(0).to(self.dtype)  # (1,H,W)
        y_delta = torch.from_numpy(y_delta).unsqueeze(0).to(self.dtype)

        if self.device is not None:
            x_gt = x_gt.to(self.device, non_blocking=True)
            x_init = x_init.to(self.device, non_blocking=True)
            y_delta = y_delta.to(self.device, non_blocking=True)

        return x_gt, x_init, y_delta


def get_lodopab_dataloader(
    init_recon: str,
    batch_size: int,
    data_root: str = "lodopab_out",
    split: str = "train",
    n_train: int = 4000,
    n_test: int = 1000,
    shuffle: bool = True,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
) -> DataLoader:
    init_recon = init_recon.lower()
    if init_recon not in ("tv", "fbp", "lw"):
        raise ValueError("init_recon must be one of: 'tv', 'fbp', 'lw'")

    dataset = LoDoPaBGTInitDataset(
        root=Path(data_root),
        init=init_recon,
        device=device,
    )

    indices = np.arange(len(dataset))
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



if __name__ == "__main__":
    import json
    summary_path = Path("lodopab_out") / "summary.json"
    with open(summary_path, "r") as f:
        summary = json.load(f)

    init = "fbp"
    BATCH_SIZE = 1
    n_train = 1
    n_test = 1
    DATA_ROOT = "lodopab_out"
    NUM_WORKERS =1
    
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