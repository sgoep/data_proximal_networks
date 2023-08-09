import torch
import torch.nn as nn
import numpy as np
from config import config
from functions.pytorch_radon import get_operators

class range_layer(nn.Module):
    def __init__(self, angles):
        super(range_layer, self).__init__()        
        self.A, self.B = get_operators(angles=angles, n_angles=len(angles), image_size=config.image_size, circle=True, device=config.device)

    def forward(self, x):
        y = self.B(self.A(x))
        return y


class null_space_layer(nn.Module):
    def __init__(self, angles):
        super(null_space_layer, self).__init__()        
        self.A, self.B = get_operators(angles=angles, n_angles=len(angles), image_size=config.image_size, circle=True, device=config.device)

    def forward(self, x):
        y = x - self.B(self.A(x))
        return y


class proximal_layer(nn.Module):
    def __init__(self, angles):
        super(proximal_layer, self).__init__()
        self.ell2_norm = torch.Tensor(np.load('data/norm2.npy', allow_pickle=True)).to(config.device)
        self.A, self.B = get_operators(angles=angles, n_angles=len(angles), image_size=config.image_size, circle=True, device=config.device)
    
    def Phi(self, x):
        y = torch.zeros_like(x)
        for i in range(x.size(0)):
            norm = torch.linalg.norm(x[i,0,:,:])
            if norm < self.ell2_norm:
                y[i] = x[i,0,:,:]
            else:
                y[i] = self.ell2_norm*x[i,0,:,:]/torch.sqrt(norm**2)
        return y
                
        return y

    def forward(self, x):
        y = self.A(x)
        y = self.Phi(y)
        z = self.B(y)
        return z
    
    