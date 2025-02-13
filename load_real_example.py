# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from misc.radon_operator import ram_lak_filter, RadonOperator
import torch
import mat73
import astra
sample = "02c"

ct_data = scipy.io.loadmat(f'data_htc2022/htc2022_test_data/htc2022_{sample}_limited.mat')
# ct_data = mat73.loadmat(f'data_htc2022/htc2022_test_data/htc2022_{sample}_limited.mat')
ct_data = ct_data["CtDataLimited"][0][0]
sino = ct_data["sinogram"]
angles = ct_data["parameters"]["angles"][0, 0][0]

r = [dict(zip(ct_data.dtype.names,x)) for x  in ct_data][2]["type"]
params = [dict(zip(r.dtype.names,x)) for x  in r]

syn_image = np.load("data_htc2022_simulated/images.npy")[1]

A = RadonOperator(angles)

# sino = ram_lak_filter(torch.Tensor(sino).to("cuda"))
# print(sino)
sino_syn = A.forward(syn_image)

sino_syn += 0.008*np.abs(np.max(sino_syn))*np.random.randn(*sino_syn.shape) 
sino_syn *= 0.00002
# astra.functions.add_noise_to_sino(sino_syn, 0.003)

fbp_syn = A.fbp(sino_syn)# * 0.00002


# fbp_lim = A.dot(sino.reshape(-1, 1)).reshape(512, 512)

# fbp = scipy.io.loadmat(f'data_htc2022/htc2022_test_data/htc2022_{sample}_recon_fbp.mat')
fbp_real = A.fbp(sino)
plt.figure()
plt.imshow(fbp_syn)
plt.colorbar()
plt.savefig("test_syn.png")

plt.figure()
plt.imshow(fbp_real)
plt.colorbar()
plt.savefig("test.png")
# %%

import numpy as np
import matplotlib.pyplot as plt
idx = 1
x = np.load(f"data_htc2022_simulated/phantom.npy")
plt.figure()
plt.imshow(x[idx])
plt.colorbar()
x = np.load(f"data_htc2022_simulated/fbp.npy")
plt.figure()
plt.imshow(x[idx])
plt.colorbar()
x = np.load(f"data_htc2022_simulated/init_regul.npy")
plt.figure()
plt.imshow(x[idx])
plt.colorbar()


# %%
import numpy as np
import matplotlib.pyplot as plt
from misc.radon_operator import ram_lak_filter
import scipy.io
import astra 

def get_real_matrix(angles):
    # Distances and pixel size in mm
    dist_src_center = 410.66
    dist_src_detector = 553.74
    pixelsize = 0.2 * dist_src_center / dist_src_detector
    # pixelsize = 0.05
    num_detectors = 560
    vol_geom_id = astra.create_vol_geom(512, 512)
    # self.angles = np.linspace(0, 2 * np.pi, 721)
    angles = angles*np.pi/180

    projection_id = astra.create_proj_geom(
        "fanflat",
        dist_src_detector / dist_src_center,
        num_detectors,
        angles,
        dist_src_center / pixelsize,
        (dist_src_detector - dist_src_center) / pixelsize)

    projector_id = astra.create_projector("cuda", projection_id, vol_geom_id)
    mat_id = astra.projector.matrix(projector_id)
    A = astra.matrix.get(mat_id)
    print("abc")
    return A

sample = "01a"

ct_data = scipy.io.loadmat(f'data_htc2022/htc2022_test_data/htc2022_{sample}_limited.mat')
# ct_data = mat73.loadmat(f'data_htc2022/htc2022_test_data/htc2022_{sample}_limited.mat')
ct_data = ct_data["CtDataLimited"][0][0]
sino = ct_data["sinogram"]
angles = ct_data["parameters"]["angles"][0, 0][0]

Amat = get_real_matrix(angles)

idx = 1
x = np.load(f"data_htc2022_simulated/phantom.npy")[idx]

A = lambda x: np.dot(Amat, x.reshape(-1, 1)).reshape(len(angles), 560)
AT = lambda y: np.dot(Amat.T, y.reshape(-1, 1)).reshape(512, 512)

g = A(x)

plt.figure()
plt.imshow(x)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(g)
plt.colorbar()
plt.show()