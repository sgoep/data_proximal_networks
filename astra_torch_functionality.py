# %%
import astra
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from misc.radon_operator import ram_lak_filter, get_matrix

# def ram_lak_filter(sinogram):
#     num_projections, num_detectors = sinogram.shape
#     freqs = torch.fft.fftfreq(num_detectors).reshape(1, -1)
#     filter = torch.abs(freqs)
#     filtered_sinogram = torch.zeros_like(sinogram)
#     for i in range(num_projections):
#         projection_fft = torch.fft.fft(sinogram[i, :])
#         filtered_projection_fft = projection_fft * filter
#         filtered_projection = torch.fft.ifft(filtered_projection_fft).real
#         filtered_sinogram[i, :] = filtered_projection * (2 / torch.sqrt(torch.tensor(num_detectors)))**2
#     return filtered_sinogram

# def get_matrix(N, Ns, al):
#     pixel_width = 1
#     volGeom = astra.create_vol_geom(N, N)
#     projGeom = astra.create_proj_geom("parallel", pixel_width, Ns, al)
#     proj_id = astra.create_projector("line", projGeom, volGeom)
#     mat_id = astra.projector.matrix(proj_id)
#     A = astra.matrix.get(mat_id)
#     return torch.Tensor(A.toarray()).to_sparse()

Nal = 180
N = 128
Ns = 200
pixel_width = 1
Phi = torch.pi / 3

al_full = torch.Tensor(np.linspace(-np.pi/2, np.pi/2*(1-1/Nal), Nal, endpoint=True))
al1 = al_full[torch.abs(al_full) <= Phi]
al2 = al_full[torch.abs(al_full) > Phi]

A1 = get_matrix(N, Ns, al1.numpy())
A2 = get_matrix(N, Ns, al2.numpy())

# f = Image.open("ncat.png").resize([N, N]).convert("L")
# f = torch.tensor(np.array(f), dtype=torch.float32)
# f = f / torch.max(f)

index = 1
f = np.load(f'./data_astra/phantom/phantom_{str(index)}.npy')
f = torch.Tensor(f)
f = f / torch.max(f)

g1 = A1.matmul(f.reshape(-1, 1)).reshape(len(al1), Ns)
g2 = A2.matmul(f.reshape(-1, 1)).reshape(len(al2), Ns)

g1 = ram_lak_filter(g1)
g2 = ram_lak_filter(g2)

fbp1 = A1.T.matmul(g1.reshape(-1, 1)).reshape(N, N)
fbp2 = A2.T.matmul(g2.reshape(-1, 1)).reshape(N, N)

plt.imshow((fbp1 + fbp2).numpy())
plt.colorbar()
plt.show()

 
