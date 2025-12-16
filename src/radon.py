# %%
import numpy as np
import math
import torch
from typing import Optional, Union, Tuple
import torch.nn.functional as F
from torch_radon import Radon
try:
    import scipy.fft
    fftmodule = scipy.fft
except ImportError:
    import numpy.fft
    fftmodule = numpy.fft


def construct_fourier_filter_torch(size: int, filter_name: str, device, dtype=torch.float32) -> torch.Tensor:
    """
    Build the Fourier-domain filter as a 1D torch tensor of shape (size,).
    """
    if size % 2 != 0:
        raise ValueError(f"size must be even, got {size}")

    filter_name = filter_name.lower()

    # Create spatial-domain impulse response f, then FFT -> frequency filter
    n = torch.cat(
        (
            torch.arange(1, size // 2 + 1, 2, device=device, dtype=torch.int64),
            torch.arange(size // 2 - 1, 0, -2, device=device, dtype=torch.int64),
        ),
        dim=0,
    )

    f = torch.zeros(size, device=device, dtype=dtype)
    f[0] = 0.25
    f[1::2] = -1.0 / (math.pi * n.to(dtype)) ** 2

    fourier_filter = 2.0 * torch.real(torch.fft.fft(f))

    if filter_name in ("ramp", "ram-lak"):
        pass

    elif filter_name == "shepp-logan":
        # omega = pi * freq, skip DC
        omega = math.pi * torch.fft.fftfreq(size, device=device, dtype=dtype)[1:]
        fourier_filter[1:] *= torch.sin(omega) / omega

    elif filter_name == "cosine":
        freq = torch.linspace(0, math.pi, size, device=device, dtype=dtype, requires_grad=False)
        cosine_filter = torch.fft.fftshift(torch.sin(freq))
        fourier_filter *= cosine_filter

    elif filter_name == "hamming":
        fourier_filter *= torch.fft.fftshift(torch.hamming_window(size, device=device, dtype=dtype))

    elif filter_name == "hann":
        fourier_filter *= torch.fft.fftshift(torch.hann_window(size, device=device, dtype=dtype))

    else:
        raise ValueError(
            f"Unknown filter type '{filter_name}'. "
            "Available: 'ramp'/'ram-lak', 'shepp-logan', 'cosine', 'hamming', 'hann'."
        )

    return fourier_filter  # (size,)


def filter_sinogram(
    Y: torch.Tensor,
    filter_name: str = "ramp",
    fourier_filter_cache: Optional[dict] = None,
) -> torch.Tensor:
    """
    Apply FBP-style 1D frequency filtering to sinograms in shape (B, C, H, W),
    filtering along W (detectors). Assumes H = angles.

    Args:
        Y: (B, C, H, W) tensor
        filter_name: filter type
        fourier_filter_cache: optional dict to cache filters by padded_size/device/dtype/name

    Returns:
        Filtered tensor with same shape as Y.
    """
    if Y.ndim != 4:
        raise ValueError(f"Expected input of shape (B, C, H, W), got {tuple(Y.shape)}")

    device = Y.device
    real_dtype = torch.float32 if Y.dtype in (torch.float16, torch.bfloat16) else Y.dtype
    B, C, n_angles, size = Y.shape

    # padded_size = max(64, next_pow2(2*size))
    padded_size = max(64, 1 << math.ceil(math.log2(2 * size)))
    pad = padded_size - size

    # Pad on the last dimension only
    Yf = F.pad(Y.to(real_dtype), (0, pad))  # (B, C, H, padded_size)

    # FFT along detector axis
    sino_fft = torch.fft.fft(Yf, dim=-1)  # complex

    # Build / cache filter
    cache_key = None
    if fourier_filter_cache is not None:
        cache_key = (padded_size, filter_name, device.type, str(device), str(real_dtype))
        f = fourier_filter_cache.get(cache_key)
    else:
        f = None

    if f is None:
        f = construct_fourier_filter_torch(padded_size, filter_name, device=device, dtype=real_dtype)
        # make complex for multiplication with FFT
        f = f.to(torch.complex64 if real_dtype == torch.float32 else torch.complex128)
        if fourier_filter_cache is not None:
            fourier_filter_cache[cache_key] = f

    # Broadcast multiply: (B,C,H,W) * (W,)
    filtered_fft = sino_fft * f.view(1, 1, 1, -1)

    # iFFT back, crop, scale
    filtered = torch.fft.ifft(filtered_fft, dim=-1).real  # (B,C,H,padded_size)
    filtered = filtered[..., :size]  # (B,C,H,W)
    filtered = filtered * (math.pi / (2.0 * n_angles))

    return filtered.to(dtype=Y.dtype)



class RadonAdapter:
    def __init__(
        self,
        resolution: int,
        angles: np.ndarray,
        det_count: int,
        clip_to_circle: bool = False,
        dataset: Union[str, None] = None,
        dx: float = 1.0,
        estimate_norm: bool = True,
        norm_iters: int = 20,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        phi: Optional[Tuple[float, float]] = None,
    ):
        self.base = Radon(
            resolution=resolution,
            angles=np.asarray(angles, dtype=np.float32),
            det_count=det_count,
            clip_to_circle=clip_to_circle,
        )
        self.dataset = (dataset or "").lower()
        self.resolution = int(resolution)
        self.det_count = int(det_count)
        self.angles = np.asarray(angles, dtype=np.float32)
        self.dx = float(dx)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self.norm_A: Optional[float] = None
        self.norm_A2: Optional[float] = None

        self.phi = phi
        self._ran_mask_np = self._build_ran_mask_np()
        self._nsn_mask_np = self._build_null_mask_np()

        self._ran_mask = torch.from_numpy(self._ran_mask_np).to(device=self.device, dtype=self.dtype)
        self._nsn_mask = torch.from_numpy(self._nsn_mask_np).to(device=self.device, dtype=self.dtype)

        if estimate_norm:
            self._estimate_operator_norm(iters=norm_iters)

    def _build_ran_mask_np(self) -> np.ndarray:
        lo, hi = self.phi
        ang_mask = ((self.angles >= lo) & (self.angles < hi)).astype(np.float32)
        mask2d = np.repeat(ang_mask.reshape(-1, 1), self.det_count, axis=1)
        return mask2d[None, None, :, :].astype(np.float32)  

    def _build_null_mask_np(self) -> np.ndarray:
        return 1.0 - self._build_ran_mask_np()

    # @torch.no_grad()
    # def _apply_phi(self, y: torch.Tensor) -> torch.Tensor:
    #     return y * self._phi_mask.to(device=y.device, dtype=y.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base.forward(x) * self.dx
        return y
        # return self._apply_phi(y)

    def forward_la(self, x: torch.Tensor) -> torch.Tensor:
        y = self.forward(x)
        return self.proj_ran(y)
    
    def backward_la(self, y: torch.Tensor) -> torch.Tensor:
        y = self.proj_ran(y)
        return self.backward(y)
    
    def backward(self, y: torch.Tensor) -> torch.Tensor:
        # y = self._apply_phi(y)
        return self.base.backward(y / self.dx)
    
    def fbp_la(self, y: torch.Tensor, filter_name: str = "ram-lak") -> torch.Tensor:
        # y = self._apply_phi(y)
        return self.backward(filter_sinogram(self.proj_ran(y), filter_name=filter_name))

    def fbp(self, y: torch.Tensor, filter_name: str = "ram-lak") -> torch.Tensor:
        # y = self._apply_phi(y)
        return self.backward(filter_sinogram(y, filter_name=filter_name))

    def proj_nsn(self, y: torch.Tensor) -> torch.Tensor:
        return y * self._nsn_mask.to(device=y.device, dtype=y.dtype)

    def proj_ran(self, y: torch.Tensor) -> torch.Tensor:
        # print(self._ran_mask.to(device=y.device, dtype=y.dtype).shape)
        return y * self._ran_mask.to(device=y.device, dtype=y.dtype)
    

    # @torch.no_grad()
    # def proj_nsn(self, x: torch.Tensor) -> torch.Tensor:
    #     y = self.forward(x) * self._nsn_mask.to(device=x.device, dtype=x.dtype)
    #     return self.fbp(y)

    # @torch.no_grad()
    # def proj_ran(self, x: torch.Tensor) -> torch.Tensor:
    #     y = self.forward(x) * self._ran_mask.to(device=x.device, dtype=x.dtype)
    #     return self.fbp(y)

    @torch.no_grad()
    def _estimate_operator_norm(
        self,
        iters: int = 20,
        tol: float = 1e-6,
        seed: int = 0,
    ) -> None:
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)

        x = torch.randn(
            (1, 1, self.resolution, self.resolution),
            device=self.device,
            dtype=self.dtype,
            generator=g,
        )
        x /= x.norm() + 1e-12

        last_lambda = None
        lam = None

        for _ in range(iters):
            y = self.forward(x)
            x_new = self.backward(y)

            lam = (x_new * x).sum().abs().item() / (x * x).sum().clamp_min(1e-12).item()
            x = x_new / (x_new.norm() + 1e-12)

            if last_lambda is not None:
                if abs(lam - last_lambda) / max(lam, 1e-12) < tol:
                    break
            last_lambda = lam

        self.norm_A2 = float(lam if lam is not None else 0.0)
        self.norm_A = float(math.sqrt(self.norm_A2))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dival.datasets import LoDoPaBDataset

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    INDEX = 0
    FILTER = "ram-lak"
    SAVE_PATH = "lodopab_gt_vs_fbp.png"

    # -------------------------------------------------
    # Load LoDoPaB sample
    # -------------------------------------------------
    dataset = LoDoPaBDataset()

    # Ground truth and measurement
    sino_np, gt_np = dataset.get_sample(INDEX, part='train')   # gt: (H,W), sino: (angles, detectors)

    gt_np = gt_np.data
    sino_np = sino_np.data
    
    gt = torch.from_numpy(gt_np).to(DEVICE).unsqueeze(0).unsqueeze(0)
    sino = torch.from_numpy(sino_np).to(DEVICE).unsqueeze(0).unsqueeze(0)

    _, _, n_angles, det_count = sino.shape
    resolution = gt.shape[-1]
    
    angles = np.linspace(-np.pi/2, np.pi/2, n_angles, endpoint=False)
    phi = (-np.pi/2, np.pi/2)

    radon = RadonAdapter(
        resolution=resolution,
        angles=angles,
        det_count=det_count,
        clip_to_circle=False,
        dx=2 * 0.13 / resolution,
        phi=phi
    )

# %%
    with torch.no_grad():
        reco_fbp = radon.fbp(sino, filter_name=FILTER)

    gt_np = gt.squeeze().cpu().numpy()
    fbp_np = reco_fbp.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(gt_np, cmap="gray")
    axes[0].set_title("LoDoPaB Ground Truth")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(fbp_np, cmap="gray")
    axes[1].set_title("FBP Reconstruction (Ram-Lak)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=200)
    plt.show()

    print(f"Saved: {SAVE_PATH}")
# # sbatch -p a6000 -w mp-gpu4-a6000-2 --job-name=radon -o logs/radon.txt --time=30-00:00:00 --wrap="python -u src/radon.py"

# # %%
