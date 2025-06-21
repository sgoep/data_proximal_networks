"""
The original code was taken from
https://github.com/matteo-ronchetti/torch-radon.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch_radon
from typing import Optional, Tuple, Union

from src.utils.load_config import load_config


try:
    import scipy.fft
    fftmodule = scipy.fft
except ImportError:
    import numpy.fft
    fftmodule = numpy.fft


def construct_fourier_filter(size: int, filter_name: str) -> np.ndarray:
    """
    Construct a Fourier domain filter for filtering sinograms.

    Args:
        size (int): The filter size (must be even).
        filter_name (str): The name of the filter to use. Supported: 
            'ramp', 'ram-lak', 'shepp-logan', 'cosine', 'hamming', 'hann'.

    Returns:
        np.ndarray: The computed Fourier filter.
    """
    filter_name = filter_name.lower()
    n = np.concatenate((
        np.arange(1, size / 2 + 1, 2, dtype=int),
        np.arange(size / 2 - 1, 0, -2, dtype=int),
    ))
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2
    fourier_filter = 2 * np.real(fftmodule.fft(f))

    if filter_name in ["ramp", "ram-lak"]:
        pass
    elif filter_name == "shepp-logan":
        omega = np.pi * fftmodule.fftfreq(size)[1:]
        fourier_filter[1:] *= np.sin(omega) / omega
    elif filter_name == "cosine":
        freq = np.linspace(0, np.pi, size, endpoint=False)
        cosine_filter = fftmodule.fftshift(np.sin(freq))
        fourier_filter *= cosine_filter
    elif filter_name == "hamming":
        fourier_filter *= fftmodule.fftshift(np.hamming(size))
    elif filter_name == "hann":
        fourier_filter *= fftmodule.fftshift(np.hanning(size))
    else:
        print(
            f"[TorchRadon] Error, unknown filter type '{filter_name}'. "
            "Available filters are: 'ramp', 'shepp-logan', 'cosine', 'hamming', 'hann'."
        )
    return fourier_filter


def _filter_sinogram(
    sinogram: torch.Tensor,
    filter_name: str = "ramp",
    fourier_filters: Optional[np.ndarray] = None
) -> torch.Tensor:
    """
    Apply 1D filtering in the frequency domain to a sinogram.

    Args:
        sinogram (torch.Tensor): Input sinogram of shape (angles, detectors).
        filter_name (str): Name of the Fourier filter to apply.
        fourier_filters (Optional[np.ndarray]): Precomputed filter (unused here).

    Returns:
        torch.Tensor: The filtered sinogram.
    """
    n_angles, size = sinogram.shape
    padded_size = max(64, int(2 ** torch.ceil(torch.log2(torch.tensor(2 * size)))))
    pad = padded_size - size
    padded_sinogram = F.pad(sinogram.float(), (0, pad))
    sino_fft = torch.fft.fft(padded_sinogram, dim=-1)

    f = construct_fourier_filter(padded_size, filter_name)
    f_tensor = torch.tensor(f, device="cuda", dtype=torch.cfloat)
    filtered_sino_fft = sino_fft * f_tensor
    filtered_sinogram = torch.fft.ifft(filtered_sino_fft, dim=-1).real
    filtered_sinogram = filtered_sinogram[:, :-pad] * (torch.pi / (2 * n_angles))
    return filtered_sinogram.to(dtype=sinogram.dtype)


def filter_sinogram(Y: torch.Tensor) -> torch.Tensor:
    """
    Batch-wise apply filtering to sinograms.

    Args:
        Y (torch.Tensor): Sinogram tensor of shape (B, 1, A, D), (1, A, D), or (A, D).

    Returns:
        torch.Tensor: Filtered sinogram with same shape as input.
    """
    if Y.ndim == 3:
        Y = Y.unsqueeze(0)
    elif Y.ndim == 2:
        Y = Y.unsqueeze(0).unsqueeze(0)

    Y_filtered = torch.zeros_like(Y).to("cuda")
    for j in range(Y.shape[0]):
        Y_filtered[j, 0, :, :] = _filter_sinogram(Y[j, 0, :, :])
    return Y_filtered


def get_radon_operator(
    N: int,
    phi: Optional[float],
    num_angles: int,
    det_count: int,
    det_spacing: float,
    clip_to_circle: bool
) -> torch_radon.Radon:
    """
    Return a Radon transform operator with optionally limited angles.

    Args:
        N (int): Image resolution.
        phi (Optional[float]): Limit of angle range (in radians).
        num_angles (int): Total number of angles to consider.
        det_count (int): Number of detector bins.
        det_spacing (float): Spacing between detector bins.
        clip_to_circle (bool): Whether to clip to circular support.

    Returns:
        torch_radon.Radon: Configured Radon transform operator.
    """
    angles = np.linspace(0, np.pi, num_angles, endpoint=False)
    if phi is not None:
        angles = angles[angles <= phi]
    return torch_radon.Radon(
        N,
        angles,
        det_count=det_count,
        det_spacing=det_spacing,
        clip_to_circle=clip_to_circle,
    )


def get_radon_operator_null_space(
    N: int,
    phi: float,
    num_angles: int,
    det_count: int,
    det_spacing: float,
    clip_to_circle: bool
) -> torch_radon.Radon:
    """
    Return a Radon transform operator for the null-space (unobserved angles).

    Args:
        N (int): Image resolution.
        phi (float): Limit angle defining the null space.
        num_angles (int): Total number of angles.
        det_count (int): Number of detector bins.
        det_spacing (float): Spacing between detector bins.
        clip_to_circle (bool): Whether to clip to circular support.

    Returns:
        torch_radon.Radon: Radon operator restricted to unobserved angles.
    """
    angles = np.linspace(0, np.pi, num_angles, endpoint=False)
    angles = angles[angles > phi]
    return torch_radon.Radon(
        N,
        angles,
        det_count=det_count,
        det_spacing=det_spacing,
        clip_to_circle=clip_to_circle,
    )


def get_radon_operators(example: str) -> Tuple[torch_radon.Radon, torch_radon.Radon, torch_radon.Radon]:
    """
    Return a triple of Radon transform operators for full, limited, and null-space angles.

    Args:
        example (str): Dataset name ("lodopab" or "synthetic").

    Returns:
        Tuple[torch_radon.Radon, torch_radon.Radon, torch_radon.Radon]: 
            full, limited, and null-space Radon operators.
    """
    config = load_config(example)

    if example == "lodopab":
        radon_full = get_radon_operator(
            N=config.N,
            phi=config.phi_full,
            num_angles=config.num_angles_full,
            det_count=config.det_count,
            det_spacing=config.det_spacing,
            clip_to_circle=config.clip_to_circle,
        )
        radon_limited = get_radon_operator(
            N=config.N,
            phi=config.phi_limited,
            num_angles=config.num_angles_full,
            det_count=config.det_count,
            det_spacing=config.det_spacing,
            clip_to_circle=config.clip_to_circle,
        )
        radon_null_space = get_radon_operator_null_space(
            N=config.N,
            phi=config.phi_limited,
            num_angles=config.num_angles_full,
            det_count=config.det_count,
            det_spacing=config.det_spacing,
            clip_to_circle=config.clip_to_circle,
        )
    elif example == "synthetic":
        radon_full = torch_radon.Radon(
            resolution=config.N,
            angles=config.angles_full,
            det_count=config.det_count,
            det_spacing=config.det_spacing,
            clip_to_circle=config.clip_to_circle,
        )
        radon_limited = torch_radon.Radon(
            resolution=config.N,
            angles=config.angles_limited,
            det_count=config.det_count,
            det_spacing=config.det_spacing,
            clip_to_circle=config.clip_to_circle,
        )
        radon_null_space = torch_radon.Radon(
            resolution=config.N,
            angles=config.angles_null_space,
            det_count=config.det_count,
            det_spacing=config.det_spacing,
            clip_to_circle=config.clip_to_circle,
        )
    else:
        raise ValueError("Example not known.")

    return radon_full, radon_limited, radon_null_space
