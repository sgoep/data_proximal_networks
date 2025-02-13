import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.utils import extension_with_zero
from src.utils.load_config import load_config
from src.utils.load_data import read_h5_file
from src.utils.projections import projection_onto_null_space, projection_onto_range
from src.utils.radon_operator import (
    filter_sinogram,
    get_radon_operator,
    get_radon_operators,
)

example = "synthetic"

config = load_config(example)

device = "cuda"
# file_name_data = "data/data_lodopabdata_original/observations/train/observation_train_000.hdf5"
file_name_gt = (
    "data/data_lodopab/data_original/ground_truth/train/ground_truth_train_000.hdf5"
)

gt = np.flipud(read_h5_file(file_name_gt)[0, :, :].T)

# gt = np.load("data/data_synthetic/phantom.npy")[0, :, :]
# self.Z = np.load("data/data_synthetic/limited_angle_data.npy")


plt.figure()
plt.imshow(gt)
plt.colorbar()
plt.savefig("main_gt.png")

# IMAGE_SHAPE = (362, 362)
# NUM_ANGLES = 1000

radon_full, radon_limited, radon_null_space = get_radon_operators(example)


gt = torch.Tensor(gt.copy()).to(device).unsqueeze(0).unsqueeze(0)

radon_full_gt = radon_full.forward(gt)
radon_limited_gt = radon_limited.forward(gt)
radon_null_space_gt = radon_null_space.forward(gt)

radon_limited_gt = radon_limited_gt.cpu().numpy()
radon_null_space_gt = radon_null_space_gt.cpu().numpy()

print(radon_limited_gt.shape)
print(radon_null_space_gt.shape)

radon_fullx = np.zeros([config.num_angles_full, config.det_count])
radon_fullx[0 : config.num_angles_limited, :] = radon_limited_gt
radon_fullx[config.num_angles_limited :, :] = radon_null_space_gt


# radon_full_gt_proj = projection_onto_null_space(radon_full_gt)

# radon_combined = extension_with_zero(radon_limited_gt) + radon_full_gt_proj
# radon_combined = projection_onto_range(radon_full_gt) + radon_full_gt_proj

# fbp_radon_combined = radon_full.backward(
#     filter_sinogram(radon_combined.squeeze())
# )

plt.figure()
plt.imshow(radon_fullx)
plt.colorbar()
plt.savefig("proj_test_1.png")

Y = torch.Tensor(radon_fullx).to("cuda")
rec = radon_full.backward(filter_sinogram(Y))

plt.figure()
plt.imshow(rec.cpu().numpy())
plt.colorbar()
plt.savefig("proj_test_2.png")

# plt.figure()
# plt.imshow(abs(radon_combined-radon_full_gt).cpu().numpy().squeeze())
# plt.colorbar()
# plt.savefig("proj_test_2.png")

# plt.figure()
# plt.imshow(radon_full_gt_proj.cpu().numpy())
# plt.colorbar()
# plt.savefig("main_radon_full_gt_proj.png")

# plt.figure()
# plt.imshow(fbp_radon_full_gt_proj.cpu().numpy())
# plt.colorbar()
# plt.savefig("main_fbp_radon_full_gt_proj.png")

# plt.figure()
# plt.imshow(radon_combined.cpu().numpy())
# plt.colorbar()
# plt.savefig("main_radon_combined.png")

# plt.figure()
# plt.imshow(fbp_radon_combined.cpu().numpy())
# plt.colorbar()
# plt.savefig("main_fbp_radon_combined.png")

print("Finished.")
