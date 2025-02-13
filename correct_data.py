from os import listdir
from os.path import isfile, join

import h5py  # type: ignore
import matplotlib.pyplot as plt
import numpy as np  # type: ignore
import torch  # type: ignore
import torch_radon
from pytorch_wavelets import DWTForward, DWTInverse

# from config import config
from src.algorithms.ell1_shearlet import ell1_shearlet
from src.algorithms.ell1_wavelet import ell1_wavelet
from src.algorithms.my_shearlet import ShearletTransform
from src.algorithms.total_variation import tv
from src.data.utils import linear_regression
from src.utils.load_config import load_config
from src.utils.load_data import read_h5_file
from src.utils.radon_operator import filter_sinogram, get_radon_operators

example = "lodopab"
which = "validation"

config = load_config(example)

radon_full, radon_limited, _ = get_radon_operators(example)

file_name_data = (
    f"data/data_lodopab/data_original/observations/{which}/observation_{which}_000.hdf5"
)

idx = 10

data = read_h5_file(file_name_data)[idx, :, :]  # * 1000
data = data[0 : config.num_angles_limited, :]

file_name_ground_truth = f"data/data_lodopab/data_original/ground_truth/{which}/ground_truth_{which}_000.hdf5"
# ground_truth = read_h5_file(file_name_ground_truth)[idx, :, :]
ground_truth = np.rot90(read_h5_file(file_name_ground_truth)[idx, :, :])
ground_truth = ground_truth / np.max(np.abs(ground_truth))
# ground_truth = np.flipud(read_h5_file(file_name_ground_truth)[0, :, :].T)

plt.figure()
plt.imshow(ground_truth, cmap="gray")
plt.colorbar()
plt.savefig("xx_ground_truth")

plt.figure()
plt.imshow(data, cmap="gray")
plt.colorbar()
plt.savefig("xx_data")

syn_data = radon_limited.forward(torch.Tensor(ground_truth.copy()).cuda())

# a, b = linear_regression(syn_data, torch.Tensor(data).cuda(), lr=0.0001, num_iterations=100)

y0 = syn_data.cpu().numpy()

# y_noise = data
# y_noise_rescaled = (y_noise - np.min(y_noise)) / (np.max(y_noise) - np.min(y_noise))
# y_noise_rescaled = y_noise_rescaled * (np.max(y0) - np.min(y0)) + np.min(y0)
# scaled_data = y_noise_rescaled
# scaled_data = torch.Tensor(scaled_data).cuda()

from scipy import stats

slope, intercept, r, p, std_err = stats.linregress(data.flatten(), y0.flatten())

scaled_data = slope * data + intercept
scaled_data = torch.Tensor(scaled_data).cuda()

fbp = (
    radon_limited.backward(filter_sinogram(torch.Tensor(scaled_data).cuda()))
    .cpu()
    .numpy()
    .squeeze()
)

plt.figure()
plt.imshow(fbp, cmap="gray")
plt.colorbar()
plt.savefig("xx_fbp")

plt.figure()
plt.imshow(syn_data.cpu().numpy().squeeze(), cmap="gray")
plt.colorbar()
plt.savefig("xx_forward_ground_truth")


plt.figure()
plt.imshow(scaled_data.cpu().numpy().squeeze(), cmap="gray")
plt.colorbar()
plt.savefig("xx_scaled_orig_data")

L = 500
Niter = 1000
alpha = 0.5
tau = 1 / L
sigma = 1 / L
theta = 1
f_tv = tv(
    torch.zeros([config.N, config.N]).cuda(),
    radon_limited,
    scaled_data,
    alpha,
    tau,
    sigma,
    theta,
    L,
    Niter,
    ground_truth=None,
    print_flag=False,
)
f = f_tv.cpu().numpy()

# shearlet = ShearletTransform(config.N, config.N, [0.5]*3)

# Niter = 50
# p_0 = 0.0005
# p_1 = 0.1
# f = ell1_shearlet(
#     shearlet=shearlet,
#     A=radon_limited,
#     sinogram=scaled_data,
#     p_0=p_0,
#     p_1=p_1,
#     Niter=Niter,
#     ground_truth=torch.Tensor(ground_truth.copy()).cuda(),
#     print_flag=True
# ).cpu().numpy()

p_0 = 0.00001
p_1 = 0.001
Niter = 100
from pytorch_wavelets import DWTForward, DWTInverse

wavelet = DWTForward(
    J=5, mode="zero", wave="db3"
).cuda()  # Accepts all wave types available to PyWavelets
iwavelet = DWTInverse(mode="zero", wave="db3").cuda()

f_ell1 = (
    ell1_wavelet(
        wavelet=wavelet,
        iwavelet=iwavelet,
        A=radon_limited,
        sinogram=scaled_data,
        p_0=p_0,
        p_1=p_1,
        Niter=Niter,
        ground_truth=None,  # torch.Tensor(ground_truth.copy()).cuda(),
        print_flag=False,
    )
    .cpu()
    .numpy()
)

plt.figure()
plt.imshow(f, cmap="gray")
plt.colorbar()
plt.savefig("xx_tv")

plt.figure()
plt.imshow(f_ell1, cmap="gray")
plt.colorbar()
plt.savefig("xx_ell1")


x0 = torch.zeros([config.N, config.N]).to("cuda")
landweber_limited = torch_radon.solvers.Landweber(
    radon_limited, projection=None, grad=False
)
landweber_rec = landweber_limited.run(x0, scaled_data, 1e-6, iterations=1000)
landweber_rec_np = landweber_rec.cpu().numpy()


plt.figure()
plt.imshow(landweber_rec_np, cmap="gray")
plt.colorbar()
plt.savefig("xx_landweber")
