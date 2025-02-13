import h5py
import numpy as np


def load_single_image(index: int = 1366) -> np.ndarray:
    images = (h5py.File("data/randshepp.mat")["data"][:]).transpose([-1, 0, 1])
    phantom = images[index, :, :]
    phantom = phantom / np.max(phantom)
    return phantom
