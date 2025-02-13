# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from misc.radon_operator import ram_lak_filter, get_matrix

def Phi(x):
    y = torch.zeros_like(x)
    ell2_norm = torch.Tensor(np.load("data_astra/norm2.npy"))
    # print(x.shape)
    # for i in range(x.size(0)):
    #     norm = torch.linalg.norm(x[i,0,:,:])
    #     if norm < self.ell2_norm:
    #         y[i] = x[i,0,:,:]
    #     else:
    #         y[i] = self.ell2_norm*x[i,0,:,:]/torch.sqrt(norm**2)
    norm = torch.linalg.norm(x)
    if norm < ell2_norm:
        y = x
    else:
        y = ell2_norm*x/torch.sqrt(norm**2)
    return y

index = 1

X = np.load(f"./data_astra/phantom/phantom_{str(index)}.npy")
N = 128
Nal = 180
Ns = 200
al_full = np.linspace(-np.pi/2, np.pi/2*(1-1/Nal), Nal, endpoint=True)
phi = np.pi/3
al1 = al_full[abs(al_full)<=phi]
A1 = get_matrix(N, Ns, al1)

X = torch.Tensor(X)[None,:,:]
g1 = A1.matmul(X.reshape(-1, 1)).reshape(len(al1), Ns)

delta = 0.0
eta   = torch.abs(g1).max()*torch.randn(*g1.shape)
noise = delta*eta
gnoise = g1 + noise


import matplotlib.pyplot as plt
fbp = A1.T.matmul(ram_lak_filter(gnoise).reshape(-1, 1)).reshape(N, N)
plt.figure()
plt.imshow(fbp)
plt.colorbar()

A = lambda x: torch.matmul(A1, x.reshape(-1, 1)).reshape(len(al1), Ns)
B = lambda y: torch.matmul(A1.T, ram_lak_filter(y).reshape(-1, 1)).reshape(128, 128)
            

y = A(X)
y = Phi(y)
z = B(y)

plt.figure()
plt.imshow(z)
plt.colorbar()
