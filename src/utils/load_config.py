from pathlib import Path

import numpy as np  # type: ignore
import torch  # type: ignore

# from example import example


class config_lodopab:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    learning_rate: float = 1e-5
    len_train: int = 1500
    len_test: int = 300
    epochs: int = 100

    training_params = {
        "learning_rate": learning_rate,
        "len_train": len_train,
        "len_test": len_test,
        "epochs": epochs,
    }

    model_params = {"batch_size": 16, "shuffle": True, "num_workers": 2}

    # Radon parameters
    num_angles_full = 1000
    N = 362
    det_count = 513
    det_spacing = 1
    clip_to_circle = False
    phi_full = None
    phi_limited = 3 * np.pi / 4
    angles_full = np.linspace(0, np.pi, num_angles_full, endpoint=False)
    angles_limited = angles_full[angles_full <= phi_limited]
    num_angles_limited = len(angles_limited)

    # angles_null_space = angles_full[angles_full > phi_limited]
    # num_angles_null_space = len(angles_null_space)
    factor = None
    if Path("data/data_lodopab/data_processed/train/norm.npy").is_file():
        norm = torch.Tensor(
            np.load("data/data_lodopab/data_processed/train/norm.npy")
        ).to(device)
    # norm = "x"


class config_synthetic:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    learning_rate: float = 1e-5
    len_train: int = 1000
    len_test: int = 200
    epochs: int = 100

    training_params = {
        "learning_rate": learning_rate,
        "len_train": len_train,
        "len_test": len_test,
        "epochs": epochs,
    }

    model_params = {"batch_size": 16, "shuffle": True, "num_workers": 2}
    # Radon parameters
    N = 128
    clip_to_circle = False

    phi_limited = 3 * np.pi / 4
    phi_full = np.pi
    det_count = 200
    det_spacing = 1
    num_angles_full = 180
    angles_full = np.linspace(0, phi_full, num_angles_full, endpoint=False)

    angles_limited = angles_full[angles_full <= phi_limited]
    num_angles_limited = len(angles_limited)

    angles_null_space = angles_full[angles_full > phi_limited]
    num_angles_null_space = len(angles_null_space)

    if Path("data/data_synthetic/norm.npy").is_file():
        norm = torch.Tensor(np.load("data/data_synthetic/norm.npy")).to(device)


class config_lotus:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N = 256
    num_detectors = 429
    det_spacing = 540 / 630
    source_distance = 2240 * 540 / 120
    clip_to_circle = False
    det_count = 429
    fan_sensor_spacing = 540 / 630

    phi_limited = 3 * np.pi / 4
    # phi_limited = np.pi
    phi_full = np.pi

    num_angles_full = 120
    angles_full = np.linspace(0, 360, num_angles_full, endpoint=False) * np.pi / 180

    angles_limited = angles_full[angles_full <= phi_limited]
    num_angles_limited = len(angles_limited)

    angles_null_space = angles_full[angles_full > phi_limited]
    num_angles_null_space = len(angles_null_space)

    if Path("data/data_synthetic/norm.npy").is_file():
        norm = torch.Tensor(np.load("data/data_synthetic/norm.npy")).to(device)


def load_config(example):

    if example == "lodopab":
        return config_lodopab
    elif example == "synthetic":
        return config_synthetic
    elif example == "lotus":
        return config_lotus
    else:
        raise ValueError(f"Unknown example: {example}")
