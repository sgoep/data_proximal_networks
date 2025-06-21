from pathlib import Path
from typing import Dict, Type, Union

import numpy as np
import torch


class config_lodopab:
    """
    Configuration class for training on the LoDoPaB dataset.
    Contains hyperparameters, Radon transform settings, and preprocessing logic.
    """

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """Device to use for computation."""

    learning_rate: float = 1e-5
    """Learning rate for training."""

    len_train: int = 1500
    """Number of training samples."""

    len_test: int = 300
    """Number of test samples."""

    epochs: int = 100
    """Number of training epochs."""

    training_params: Dict[str, float] = {
        "learning_rate": learning_rate,
        "len_train": len_train,
        "len_test": len_test,
        "epochs": epochs,
    }
    """Dictionary of training-related hyperparameters."""

    model_params: Dict[str, Union[int, bool]] = {
        "batch_size": 16,
        "shuffle": True,
        "num_workers": 2,
    }
    """Parameters for the DataLoader."""

    # Radon transform parameters
    num_angles_full: int = 1000
    N: int = 362
    det_count: int = 513
    det_spacing: float = 1.0
    clip_to_circle: bool = False
    phi_full: Union[float, None] = None
    phi_limited: float = 3 * np.pi / 4

    angles_full: np.ndarray = np.linspace(0, np.pi, num_angles_full, endpoint=False)
    """Full set of projection angles."""

    angles_limited: np.ndarray = angles_full[angles_full <= phi_limited]
    """Limited angle subset."""

    num_angles_limited: int = len(angles_limited)
    """Number of limited angles."""

    factor: Union[float, None] = None
    """Optional scaling factor (can be learned or fixed)."""

    if Path("data/data_lodopab/data_processed/train/norm.npy").is_file():
        norm: torch.Tensor = torch.Tensor(
            np.load("data/data_lodopab/data_processed/train/norm.npy")
        ).to(device)
        """Precomputed normalization tensor."""
    else:
        norm: torch.Tensor | None = None


class config_synthetic:
    """
    Configuration class for training on a synthetic dataset.
    Includes training hyperparameters and Radon transform setup.
    """

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """Device to use for computation."""

    learning_rate: float = 1e-5
    len_train: int = 1000
    len_test: int = 200
    epochs: int = 100

    training_params: Dict[str, float] = {
        "learning_rate": learning_rate,
        "len_train": len_train,
        "len_test": len_test,
        "epochs": epochs,
    }

    model_params: Dict[str, Union[int, bool]] = {
        "batch_size": 16,
        "shuffle": True,
        "num_workers": 2,
    }

    # Radon transform parameters
    N: int = 128
    clip_to_circle: bool = False
    phi_limited: float = 3 * np.pi / 4
    phi_full: float = np.pi
    det_count: int = 200
    det_spacing: float = 1.0
    num_angles_full: int = 180

    angles_full: np.ndarray = np.linspace(0, phi_full, num_angles_full, endpoint=False)
    angles_limited: np.ndarray = angles_full[angles_full <= phi_limited]
    num_angles_limited: int = len(angles_limited)
    angles_null_space: np.ndarray = angles_full[angles_full > phi_limited]
    num_angles_null_space: int = len(angles_null_space)

    factor: Union[float, None] = None

    if Path("data/data_synthetic/norm.npy").is_file():
        norm: torch.Tensor = torch.Tensor(np.load("data/data_synthetic/norm.npy")).to(
            device
        )
    else:
        norm: torch.Tensor | None = None


def load_config(example: str) -> Type[Union[config_lodopab, config_synthetic]]:
    """
    Load a configuration class based on the dataset example name.

    Args:
        example (str): One of "lodopab", "synthetic", or "lotus".

    Returns:
        Type[config_lodopab | config_synthetic]: Corresponding configuration class.

    Raises:
        ValueError: If the example name is unknown.
    """
    if example == "lodopab":
        return config_lodopab
    elif example == "synthetic":
        return config_synthetic
    elif example == "lotus":
        return config_lotus  # You need to define `config_lotus` elsewhere
    else:
        raise ValueError(f"Unknown example: {example}")
