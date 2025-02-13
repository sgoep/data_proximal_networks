# %%
import numpy as np
import astra
import torch
from config import config

def ram_lak_filter_np(sinogram):
    num_projections, num_detectors = sinogram.shape
    freqs = np.fft.fftfreq(num_detectors).reshape(1, -1)
    filter = np.abs(freqs)
    filtered_sinogram = np.zeros_like(sinogram)
    for i in range(num_projections):
        projection_fft = np.fft.fft(sinogram[i, :])
        filtered_projection_fft = projection_fft * filter
        filtered_projection = np.fft.ifft(filtered_projection_fft).real
        filtered_sinogram[i, :] = filtered_projection * (2 / np.sqrt(num_detectors))**2
    return filtered_sinogram

def ram_lak_filter(sinogram):
    num_projections, num_detectors = sinogram.shape
    freqs = torch.fft.fftfreq(num_detectors).reshape(1, -1)
    filter = torch.abs(freqs).to(config.device)
    filtered_sinogram = torch.zeros_like(sinogram).to(config.device)
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
    # return A.toarray()


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

    projector_id = astra.create_projector("line_fanflat", projection_id, vol_geom_id)
    mat_id = astra.projector.matrix(projector_id)
    A = astra.matrix.get(mat_id)
    return A.toarray()

class RadonOperator:
    def __init__(self, angles):
        # Distances and pixel size in mm
        dist_src_center = 410.66
        dist_src_detector = 553.74
        pixelsize = 0.2 * dist_src_center / dist_src_detector
        # pixelsize = 0.05
        self.num_detectors = 560
        self.vol_geom_id = astra.create_vol_geom(512, 512)
        # self.angles = np.linspace(0, 2 * np.pi, 721)
        self.angles = angles*np.pi/180

        self.projection_id = astra.create_proj_geom(
            "fanflat",
            dist_src_detector / dist_src_center,
            self.num_detectors,
            self.angles,
            dist_src_center / pixelsize,
            (dist_src_detector - dist_src_center) / pixelsize)
        # self.projection_id = astra.create_proj_geom(
        #     "cone",
        #     "det_row_count": 2368,
        #     "det_col_count": 2240,
        #     "angles": angles,
        # )

        projector_id = astra.create_projector("cuda", self.projection_id, self.vol_geom_id)

        volume_id = astra.data2d.create("-vol", self.vol_geom_id)

        self.volume_id = volume_id
        self.projector_id = projector_id

    def forward(self, image):
        # image_copy = image.copy()
        # image_copy = image.clone().detach()
        if not isinstance(image, np.ndarray):
            image = image.clone().detach().cpu().numpy()
        astra.data2d.store(self.volume_id, image)
        sinogram_id, sinogram = astra.create_sino(self.volume_id, self.projector_id)
        # Release memory
        astra.data2d.delete(sinogram_id)
        return sinogram
    
    def backward(self, data):
        id, backproj = astra.create_backprojection(data, self.projector_id, returnData=True)
        astra.data2d.delete(id)
        return backproj
    
    def fbp(self, data):
        id, fbp = astra.creators.create_reconstruction("FBP_CUDA", self.projector_id, data, returnData=True, filterType="ram-lak")
        astra.data2d.delete(id)
        return fbp


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