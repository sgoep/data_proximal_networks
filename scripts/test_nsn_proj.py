import numpy as np
import matplotlib.pyplot as plt
import torch

from src.utils import create_simple_phantom
from src.radon import RadonAdapter
import matplotlib.pyplot as plt
from dival.datasets import EllipsesDataset

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
rec = radon.fbp(data)# + radon_nsn.fbp(radon_nsn.forward(x))

rec_nsn = rec + radon.fbp(radon.proj_nsn(radon.forward(x)))#(x - radon.fbp(radon.forward(x)))

# rec should be (B,C,H,W); plot first image
rec_img = rec[0, 0].detach().cpu().numpy()
# data_img = radon_ran.forward(x)[0, 0].detach().cpu().numpy()
# data_nsn = radon_nsn.forward(x)[0, 0].detach().cpu().numpy()
rec_nsn_img = rec_nsn[0, 0].detach().cpu().numpy()
# rec_sum = radon_ran.forward(x) + radon_nsn.forward(x)

print("Plotting.")
plt.figure(figsize=(5, 4))
plt.subplot(1, 2, 1)
plt.imshow(rec_img, cmap="gray")
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(rec_nsn_img, cmap="gray")
plt.colorbar()
# plt.subplot(1, 3, 3)
# plt.imshow(data_nsn, cmap="gray")
# plt.colorbar()
plt.tight_layout()
plt.savefig("test_nsn.png", dpi=150)
plt.close()

print("Finished.")

# print("Create RadonAdapter and data.")
# radon_ran = RadonAdapter(
#     resolution=128,
#     angles=angles,
#     det_count=det_count,
#     clip_to_circle=False,
#     dx=1.0,
#     estimate_norm=False,
#     device=device,
#     dtype=torch.float32,
#     phi=(0, np.pi/2)
# )

# radon_nsn = RadonAdapter(
#     resolution=128,
#     angles=angles,
#     det_count=128,
#     clip_to_circle=False,
#     dx=1.0,
#     estimate_norm=False,
#     device=device,
#     dtype=torch.float32,
#     phi=(np.pi/2, np.pi)
# )

# radon_full = RadonAdapter(
#     resolution=128,
#     angles=angles,
#     det_count=128,
#     clip_to_circle=False,
#     dx=1.0,
#     estimate_norm=False,
#     device=device,
#     dtype=torch.float32,
#     phi=(-np.pi/2, np.pi/2)
# )

# data = radon_ran.forward(x)
# noise = torch.randn_like(data)
# data = data + 0.03 * data.abs().max() * noise
# rec = radon_ran.fbp(data)# + radon_nsn.fbp(radon_nsn.forward(x))
# rec_nsn = rec + (x - radon_ran.fbp(radon_ran.forward(x)))

# # rec should be (B,C,H,W); plot first image
# rec_img = rec[0, 0].detach().cpu().numpy()
# data_img = radon_ran.forward(x)[0, 0].detach().cpu().numpy()
# data_nsn = radon_nsn.forward(x)[0, 0].detach().cpu().numpy()
# rec_nsn_img = rec_nsn[0, 0].detach().cpu().numpy()
# # rec_sum = radon_ran.forward(x) + radon_nsn.forward(x)

# print("Plotting.")
# plt.figure(figsize=(5, 4))
# plt.subplot(1, 2, 1)
# plt.imshow(rec_img, cmap="gray")
# plt.colorbar()
# plt.subplot(1, 2, 2)
# plt.imshow(rec_nsn_img, cmap="gray")
# plt.colorbar()
# # plt.subplot(1, 3, 3)
# # plt.imshow(data_nsn, cmap="gray")
# # plt.colorbar()
# plt.title("TV Reconstruction (RadonAdapter)")
# plt.tight_layout()
# plt.savefig("test_nsn.png", dpi=150)
# plt.close()

# print("Finished.")

# sbatch -p a6000 -w mp-gpu4-a6000-3 --job-name=test_nsn -o logs/test_nsn.txt --time=30-00:00:00 --wrap="python -u -m scripts.test_nsn_proj"
