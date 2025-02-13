import numpy as np
import torch

from example import example
from src.utils.load_config import load_config

config = load_config(example)


def projection_onto_null_space(y: torch.Tensor) -> torch.Tensor:

    # angles = np.linspace(0, np.pi, config.num_angles_full, endpoint=False)
    angles = config.angles_limited
    num_angles = len(angles)

    chi = torch.ones(config.num_angles_full, config.det_count).to(config.device)
    chi[0:num_angles, :] = 0

    y_proj = torch.zeros_like(y).to(config.device)
    for j in range(y_proj.shape[0]):
        y_proj[j, 0, :, :] = y[j, 0, :, :] * chi
    return y_proj


def projection_onto_range(y: torch.Tensor) -> torch.Tensor:
    return y - projection_onto_null_space(y)
