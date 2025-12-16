import numpy as np
import matplotlib.pyplot as plt
import torch

from src.utils import create_simple_phantom
from src.radon import RadonAdapter
import matplotlib.pyplot as plt
from dival.datasets import EllipsesDataset

@torch.no_grad()
def make_feasible_iters(radon, x0, y, beta, alpha=1e-4, iters=100):
    x = x0
    for _ in range(iters):
        r = radon.forward_la(x) - y
        B = r.shape[0]
        n = torch.linalg.norm(r.view(B, -1), dim=1).clamp_min(1e-12)
        if torch.all(n <= beta):
            break
        scale = torch.minimum(torch.ones_like(n), beta / n).view(B, 1, 1, 1)
        r_shr = r * scale
        x = x - alpha * radon.backward_la(r - r_shr)
    return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Load image.")
# x = create_simple_phantom(128, device=device).to(device)  # (H,W) or (1,1,H,W) depending on your helper
# if x.ndim == 2:
#     x = x.unsqueeze(0).unsqueeze(0)  # -> (1,1,H,W)

N = 128

dataset = EllipsesDataset(image_size=N)
gen = dataset.generator("train")
x = torch.from_numpy(next(gen).data).to(device)

angles = np.arange(-90, 90) * np.pi/180

det_count = int(np.sqrt(2)*N)+1

radon = RadonAdapter(
    resolution=N,
    angles=angles,
    det_count=det_count,
    clip_to_circle=False,
    dx=1.0,
    estimate_norm=False,
    device=device,
    dtype=torch.float32,
    phi=(-np.pi/3, np.pi/3)
)

data = radon.forward_la(x)
noise = torch.randn_like(data)
data = data + 0.03 * data.abs().max() * radon.proj_ran(noise)
fbp = radon.fbp(data).squeeze().detach().cpu().numpy()# + radon_nsn.fbp(radon_nsn.forward(x))
beta = torch.linalg.norm(0.03 * data.abs().max() * radon.proj_ran(noise))

x0 = torch.zeros_like(x)
rec = make_feasible_iters(radon, x0, data, beta)
x = x.detach().cpu().numpy()
rec = rec[0, 0].detach().cpu().numpy()


print("Plotting.")
plt.figure(figsize=(5, 4))
plt.subplot(1, 3, 1)
plt.imshow(x, cmap="gray")
plt.colorbar()
plt.subplot(1, 3, 2)
plt.imshow(fbp, cmap="gray")
plt.colorbar()
plt.subplot(1, 3, 3)
plt.imshow(rec, cmap="gray")
plt.colorbar()
plt.tight_layout()
plt.savefig("test_ran.png", dpi=150)
plt.close()

print("Finished.")

# sbatch -p a6000 -w mp-gpu4-a6000-3 --job-name=test_ran -o logs/test_ran.txt --time=30-00:00:00 --wrap="python -u -m scripts.test_ran_proj"
