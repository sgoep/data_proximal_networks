# %%
import torch
import torch.nn as nn
import numpy as np
from config import config

class range_layer(nn.Module):
    def __init__(self):
        super(range_layer, self).__init__()        
        # self.A, self.B = get_operators(angles=angles, n_angles=len(angles), image_size=config.image_size, circle=True, device=config.device)

    def forward(self, x, A, B):
        # y = self.B(self.A(x))
        Z = torch.zeros_like(x)
        
        for j in range(x.shape[0]):
            Z[j,0,:,:] = torch.Tensor(B(A(x[j,0,:,:])))
        return Z
        # return y


class null_space_layer(nn.Module):
    def __init__(self):
        super(null_space_layer, self).__init__()        

    def forward(self, x, A, B):
        Z = torch.zeros_like(x)
        for j in range(x.shape[0]):
            Z[j,0,:,:] = x[j,0,:,:] - torch.Tensor(B(A(x[j,0,:,:]))).to(config.device)
        return Z
        # y = x - self.B(self.A(x))
        # return y


class proximal_layer(nn.Module):
    def __init__(self):
        super(proximal_layer, self).__init__()
        # self.ell2_norm = torch.Tensor(np.load('data_htc2022_simulated/norm2.npy', allow_pickle=True)).to(config.device)
        self.ell2_norm = np.load('data_htc2022_simulated/norm2.npy', allow_pickle=True)
        # self.A, self.B = get_operators(angles=angles, n_angles=len(angles), image_size=config.image_size, circle=True, device=config.device)

    def Phi(self, x):
        # y = torch.zeros_like(x)
        y = np.zeros_like(x)
        # print(x.shape)
        # for i in range(x.size(0)):
        #     norm = torch.linalg.norm(x[i,0,:,:])
        #     if norm < self.ell2_norm:
        #         y[i] = x[i,0,:,:]
        #     else:
        #         y[i] = self.ell2_norm*x[i,0,:,:]/torch.sqrt(norm**2)
        # norm = torch.linalg.norm(x)
        norm = np.linalg.norm(x)
        if norm < self.ell2_norm:
            y = x
        else:
            y = self.ell2_norm*x/np.sqrt(norm**2)
        return y
                

    def forward(self, x, A, B):
        Z = torch.zeros_like(x)
        for j in range(x.shape[0]):
            y = A(x[j,0,:,:])
            y = self.Phi(y)
            z = B(y)
            Z[j,0,:,:] = torch.Tensor(z)
        return Z
