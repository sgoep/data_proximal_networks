import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_radon

from src.utils.radon_operator import filter_sinogram

img_size = 128
det_count = 201
det_spacing = 1
num_angles = 180
angles = np.linspace(0, np.pi, num_angles, endpoint=False)

num_angles = len(angles)

radon = torch_radon.Radon(
    img_size,
    angles,
    det_count=det_count,
    det_spacing=det_spacing,
    clip_to_circle=False,
)

x = np.zeros([img_size, img_size])
x[img_size // 4 : 3 * img_size // 4, img_size // 4 : 3 * img_size // 4] = 1

x = torch.tensor(x, dtype=torch.float32, device="cuda")

data = radon.forward(x)
data_filt = filter_sinogram(data)
fbp = radon.backward(data_filt)

fbp_scaled = radon.backward(10 * data_filt)

fbp = fbp.squeeze()
fbp_scaled = fbp_scaled.squeeze()


fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Plot each image
cax1 = axes[0].imshow(x.squeeze().cpu().numpy(), cmap="gray")
axes[0].set_title("Image 1")
axes[0].axis("off")
fig.colorbar(cax1, ax=axes[0], orientation="vertical")

cax2 = axes[1].imshow(fbp.cpu().numpy(), cmap="gray")
axes[1].set_title("Image 2")
axes[1].axis("off")
fig.colorbar(cax2, ax=axes[1], orientation="vertical")

cax3 = axes[2].imshow(fbp_scaled.cpu().numpy(), cmap="gray")
axes[2].set_title("Image 3")
axes[2].axis("off")
fig.colorbar(cax3, ax=axes[2], orientation="vertical")

# Show the plot
plt.tight_layout()
plt.savefig("linearity_test.png")
