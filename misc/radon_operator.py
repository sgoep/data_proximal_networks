# %%
import numpy as np
import astra
import torch

def ram_lak_filter(sinogram):
    num_projections, num_detectors = sinogram.shape
    freqs = torch.fft.fftfreq(num_detectors).reshape(1, -1)
    filter = torch.abs(freqs)
    filtered_sinogram = torch.zeros_like(sinogram)
    for i in range(num_projections):
        projection_fft = torch.fft.fft(sinogram[i, :])
        filtered_projection_fft = projection_fft * filter
        filtered_projection = torch.fft.ifft(filtered_projection_fft).real
        filtered_sinogram[i, :] = filtered_projection * (2 / torch.sqrt(torch.tensor(num_detectors)))**2
    return filtered_sinogram

def get_matrix(N, Ns, al):
    pixel_width = 1
    volGeom = astra.create_vol_geom(N, N)
    projGeom = astra.create_proj_geom("parallel", pixel_width, Ns, al)
    proj_id = astra.create_projector("line", projGeom, volGeom)
    mat_id = astra.projector.matrix(proj_id)
    A = astra.matrix.get(mat_id)
    return torch.Tensor(A.toarray()).to_sparse()

# def filter_sinogram_torch(g):
#     a, b = g.shape
#     # Create the filter using torch.linspace and torch.cat
#     filter = torch.cat((torch.Tensor(np.linspace(1, 0, b//2, endpoint=False)), 
#                         torch.Tensor(np.linspace(0, 1, b//2, endpoint=False))))
#     filter = filter[None, :].to(g.device)
    
#     # Initialize gfilt tensor with the same shape and type as g, but complex
#     gfilt = torch.zeros_like(g, dtype=torch.complex64)
    
#     for i in range(a):
#         ghat = torch.fft.fftshift(torch.fft.fft(g[i, :])) / g.shape[0]
#         gfilt[i, :] = torch.fft.ifft(torch.fft.ifftshift(ghat * filter))
    
#     return torch.real(gfilt)

# def filter_sinogram(g):
#     a, b = g.shape
#     # max_al = max(al)
#     # filter = np.append( np.linspace(max_al, 0, b//2, endpoint=False), np.linspace(0, max_al, b//2, endpoint=False) )
#     filter = np.append( np.linspace(1, 0, b//2, endpoint=False), np.linspace(0, 1, b//2, endpoint=False) )
#     filter = filter[None,:]
#     gfilt = np.zeros_like(g, dtype="complex")
#     for i in range(a):
#         ghat = np.fft.fftshift(np.fft.fft(g[i,:]))/(g.shape[0])
#         gfilt[i,:] = np.fft.ifft(np.fft.ifftshift(ghat* filter))# * filter)
#     return np.real(gfilt)

# def get_radon_operator(N1, N2, Ns, al, pixel_width=1):
#     volumeGeometry = astra.create_vol_geom(N1, N2)
#     projectionGeometry = astra.create_proj_geom('parallel', pixel_width, Ns, al)
#     proj_id = astra.create_projector('line', projectionGeometry, volumeGeometry)
#     A = astra.OpTomo(proj_id)
#     return A, proj_id


# def get_radon_matrix(N1, N2, Ns, al, pixel_width=1):
#     _, proj_id = get_radon_operator(N1, N2, Ns, al, pixel_width=1)
#     mat_id = astra.projector.matrix(proj_id)
#     Amat = astra.matrix.get(mat_id)
#     return Amat

# index = 1
# X = np.load(f'../data/phantom/phantom_{str(index)}.npy')

# Ns = 200
# Nal = 180
# Phi = np.pi/4

# al_full = np.linspace(-np.pi/2, np.pi/2, Nal, endpoint=False)
# al1 = al_full[np.abs(al_full)<=Phi]
# al2 = al_full[np.abs(al_full)>Phi]

# # phi = np.pi/4
# # al1 = np.linspace(0, np.pi-phi, Nal, endpoint=False)
# # al2 = np.linspace(phi, np.pi, Nal, endpoint=False)

# Nal1 = len(al1)
# Nal2 = len(al2)

# A1 = get_radon_matrix(X.shape[0], X.shape[1], Ns, al1)
# A2 = get_radon_matrix(X.shape[0], X.shape[1], Ns, al2)

# import matplotlib.pyplot as plt

# plt.figure()
# plt.imshow(X)

# g1 = A1.dot(X.reshape(-1, 1)).reshape(Nal1, Ns)
# g2 = A2.dot(X.reshape(-1, 1)).reshape(Nal2, Ns)

# plt.figure()
# plt.imshow(g1)

# gfilt1 = filter_sinogram(g1)#, al1)
# gfilt2 = filter_sinogram(g2)#, al2)

# rec1 = A1.T.dot(gfilt1.reshape(-1, 1)).reshape(X.shape[0], X.shape[1])
# rec2 = A2.T.dot(gfilt2.reshape(-1, 1)).reshape(X.shape[0], X.shape[1])

# plt.figure()
# plt.imshow(rec1 + rec2)
# plt.colorbar()

# # %%

# index = 1
# X = np.load(f'../data/phantom/phantom_{str(index)}.npy')

# Ns = 200
# Nal = 180
# Phi = np.pi/2
# al_full = np.linspace(-np.pi/2, np.pi/2, Nal, endpoint=False)
# al1 = al_full[np.abs(al_full)<=Phi]

# Nal1 = len(al1)

# A1 = get_radon_matrix(X.shape[0], X.shape[1], Ns, al1)

# import matplotlib.pyplot as plt

# plt.figure()
# plt.imshow(X)

# g1 = A1.dot(X.reshape(-1, 1)).reshape(Nal1, Ns)

# plt.figure()
# plt.imshow(g1)

# gfilt1 = filter_sinogram(g1, al1)

# rec1 = A1.T.dot(gfilt1.reshape(-1, 1)).reshape(X.shape[0], X.shape[1])

# plt.figure()
# plt.imshow(rec1)
# plt.colorbar()