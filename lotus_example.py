# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch_radon
import torch_radon.solvers
from pytorch_wavelets import DWTForward, DWTInverse

from src.algorithms.ell1_wavelet import ell1_wavelet
from src.algorithms.total_variation import tv
from src.utils.radon_operator import filter_sinogram

f = scipy.io.loadmat("data/LotusData256.mat")

data = f["m"].T

wavelet = DWTForward(J=5, mode="zero", wave="db3").cuda()
iwavelet = DWTInverse(mode="zero", wave="db3").cuda()


angles = np.linspace(0, 360, data.shape[0], endpoint=True)
angles = angles * np.pi / 180

angles = angles[angles <= 3 * np.pi / 4]

source_distance = 2240 * 540 / 120
origin_detector_distance = 630
fan_sensor_spacing = 540 / 630
num_detectors = data.shape[1]

N = 256

data = torch.Tensor(data).cuda()
data = data[0 : len(angles), :]

A = torch_radon.RadonFanbeam(
    N,
    angles,
    source_distance,
    det_count=num_detectors,
    det_spacing=fan_sensor_spacing,
    clip_to_circle=False,
)

rec = A.backward(filter_sinogram(data))

landweber = torch_radon.solvers.Landweber(A, projection=None, grad=False)

x0 = torch.zeros([N, N]).cuda()

rec = landweber.run(x0, data, 1e-6, iterations=10000)

L = 500
Niter = 1000
alpha = 0.005
tau = 1 / L
sigma = 1 / L
theta = 1
f_tv = tv(x0, A, data, alpha, tau, sigma, theta, L, Niter, ground_truth=None, print_flag=False)

p_0 = 0.0005
p_1 = 0.05
Niter = 100

wavelet = DWTForward(J=5, mode="zero", wave="db3").cuda()
iwavelet = DWTInverse(mode="zero", wave="db3").cuda()

f_ell1 = ell1_wavelet(
    wavelet,
    iwavelet,
    A,
    data,
    p_0=0.00009,
    p_1=0.00001,
    Niter=500,
    ground_truth=None,
    print_flag=False,
)

plt.figure()
plt.imshow(rec.cpu().numpy().squeeze())
# plt.imshow(parallel_proj_data.cpu().numpy())
plt.colorbar()
plt.savefig("test_lotus.png")

plt.figure()
plt.imshow(f_tv.cpu().numpy().squeeze())
# plt.imshow(parallel_proj_data.cpu().numpy())
plt.colorbar()
plt.savefig("test_tv_lotus.png")

plt.figure()
plt.imshow(f_ell1.cpu().numpy().squeeze())
# plt.imshow(parallel_proj_data.cpu().numpy())
plt.colorbar()
plt.savefig("test_ell1_lotus.png")
