# %%
import torch
import torch.nn as nn
import numpy as np
from config import config
from functions.pytorch_radon import get_operators
from misc.radon_operator import filter_sinogram_torch
import astra

def get_radon_operator(N1, N2, Ns, al, pixel_width=1):
    volumeGeometry = astra.create_vol_geom(N1, N2)
    projectionGeometry = astra.create_proj_geom('parallel', pixel_width, Ns, al)
    proj_id = astra.create_projector('line', projectionGeometry, volumeGeometry)
    A = astra.OpTomo(proj_id)
    return A, proj_id, volumeGeometry, projectionGeometry


def get_radon_matrix(N1, N2, Ns, al, pixel_width=1):
    _, proj_id, _, _ = get_radon_operator(N1, N2, Ns, al, pixel_width=1)
    mat_id = astra.projector.matrix(proj_id)
    Amat = astra.matrix.get(mat_id)
    return Amat

class range_layer(nn.Module):
    def __init__(self):
        super(range_layer, self).__init__()        
        # self.A, self.B = get_operators(angles=angles, n_angles=len(angles), image_size=config.image_size, circle=True, device=config.device)
        Nal = 180
        Phi = np.pi/4
        al_full = np.linspace(-np.pi/2, np.pi/2, Nal, endpoint=False)
        al1 = al_full[np.abs(al_full)<=Phi]
        al2 = al_full[np.abs(al_full)>Phi]

        Ns = 200
        Am = torch.Tensor(get_radon_matrix(128, 128, Ns, al1).toarray()).to("cuda")
        Bm = torch.Tensor(get_radon_matrix(128, 128, Ns, al2).toarray()).to("cuda")
        
        self.A = lambda x: torch.matmul(Am, x.reshape(-1, 1)).reshape(len(al1), Ns)
        self.B = lambda y: torch.matmul(Bm.T, filter_sinogram_torch(y).reshape(-1, 1)).reshape(128, 128)

    def forward(self, x):
        y = self.B(self.A(x))
        return y


class null_space_layer(nn.Module):
    def __init__(self):
        super(null_space_layer, self).__init__()        
        # self.A, self.B = get_operators(angles=angles, n_angles=len(angles), image_size=config.image_size, circle=True, device=config.device)
        Nal = 180
        Phi = np.pi/4
        al_full = np.linspace(-np.pi/2, np.pi/2, Nal, endpoint=False)
        al1 = al_full[np.abs(al_full)<=Phi]
        al2 = al_full[np.abs(al_full)>Phi]

        Ns = 200
        Am = torch.Tensor(get_radon_matrix(128, 128, Ns, al1).toarray()).to("cuda")
        Bm = torch.Tensor(get_radon_matrix(128, 128, Ns, al2).toarray()).to("cuda")
        
        self.A = lambda x: torch.matmul(Am, x.reshape(-1, 1)).reshape(len(al1), Ns)
        self.B = lambda y: torch.matmul(Bm.T, filter_sinogram_torch(y).reshape(-1, 1)).reshape(128, 128)

    def forward(self, x):
        y = x - self.B(self.A(x))
        return y


class proximal_layer(nn.Module):
    def __init__(self):
        super(proximal_layer, self).__init__()
        self.ell2_norm = torch.Tensor(np.load('data/norm2.npy', allow_pickle=True)).to(config.device)
        # self.A, self.B = get_operators(angles=angles, n_angles=len(angles), image_size=config.image_size, circle=True, device=config.device)

        Nal = 180
        Phi = np.pi/4
        al_full = np.linspace(-np.pi/2, np.pi/2, Nal, endpoint=False)
        al1 = al_full[np.abs(al_full)<=Phi]
        al2 = al_full[np.abs(al_full)>Phi]

        Ns = 200
        Am = torch.Tensor(get_radon_matrix(128, 128, Ns, al1).toarray()).to("cuda")
        Bm = torch.Tensor(get_radon_matrix(128, 128, Ns, al2).toarray()).to("cuda")
        
        self.A = lambda x: torch.matmul(Am, x.reshape(-1, 1)).reshape(len(al1), Ns)
        self.B = lambda y: torch.matmul(Bm.T, filter_sinogram_torch(y).reshape(-1, 1)).reshape(128, 128)
            
    def Phi(self, x):
        y = torch.zeros_like(x)
        for i in range(x.size(0)):
            norm = torch.linalg.norm(x[i,0,:,:])
            if norm < self.ell2_norm:
                y[i] = x[i,0,:,:]
            else:
                y[i] = self.ell2_norm*x[i,0,:,:]/torch.sqrt(norm**2)
        return y
                

    def forward(self, x):
        y = self.A(x)
        y = self.Phi(y)
        z = self.B(y)
        return z
    

        

from misc.data_loader import DataLoader
# Y = next(iter(DataLoader)) 
counter = 1
for X, Y in DataLoader(ID=1, CONSTRAINT=False, INITRECON=False):
    if counter == 1:
        break

def filter_sinogram_torch(g):
    a, b = g.shape
    # Create the filter using torch.linspace and torch.cat
    filter = torch.cat((torch.Tensor(np.linspace(1, 0, b//2, endpoint=False)), 
                        torch.Tensor(np.linspace(0, 1, b//2, endpoint=False))))
    filter = filter[None, :].to(g.device)
    
    # Initialize gfilt tensor with the same shape and type as g, but complex
    gfilt = torch.zeros_like(g, dtype=torch.complex64)
    
    for i in range(a):
        ghat = torch.fft.fftshift(torch.fft.fft(g[i, :])) / g.shape[0]
        gfilt[i, :] = torch.fft.ifft(torch.fft.ifftshift(ghat * filter))
    
    return torch.real(gfilt)

X = torch.Tensor(X)
Y = torch.Tensor(Y)

Nal = 180
Phi = np.pi/4
al_full = np.linspace(-np.pi/2, np.pi/2, Nal, endpoint=False)
al1 = al_full[np.abs(al_full)<=Phi]
al2 = al_full[np.abs(al_full)>Phi]

Ns = 200
Am = torch.Tensor(get_radon_matrix(128, 128, Ns, al1).toarray())
Bm = torch.Tensor(get_radon_matrix(128, 128, Ns, al2).toarray())

A = lambda x: torch.matmul(Am, x.reshape(-1, 1)).reshape(len(al1), Ns)
B = lambda y: torch.matmul(Bm.T, filter_sinogram_torch(y).reshape(-1, 1)).reshape(128, 128)

g = A(X)
g = filter_sinogram_torch(g)

print(Y)

# %%
