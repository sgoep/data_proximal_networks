# %%
import numpy as np
import astra
import torch
import os 
from config import config
import pandas as pd
from misc.radon_operator import RadonOperator, get_matrix, get_real_matrix
import scipy.io
import matplotlib.pyplot as plt

from total_variation_regularization_astra import tv as tv_torch

from skimage.metrics import structural_similarity   as ssim
from skimage.metrics import mean_squared_error      as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np

def my_grad(X):
    fx = np.concatenate((X[1:,:],np.expand_dims(X[-1,:], axis=0)), axis=0) - X
    fy = np.concatenate((X[:,1:],np.expand_dims(X[:,-1], axis=1)), axis=1) - X
    return fx, fy

def my_div(Px, Py):
    fx = Px - np.concatenate((np.expand_dims(Px[0,:], axis=0), Px[0:-1,:]), axis=0)
    fx[0,:] = Px[0,:]
    fx[-1,:] = -Px[-2,:]
   
    fy = Py - np.concatenate((np.expand_dims(Py[:,0], axis=1), Py[:,0:-1]), axis=1)
    fy[:,0] = Py[:,0]
    fy[:,-1] = -Py[:,-2]

    return fx + fy


def tv(x0, A, AT, g, alpha, L, Niter, f=None, print_flag=True):
    
    tau = 1/L
    sigma = 1/L
    theta = 1
 
    grad_scale = 1e+2
    m, n    = x0.shape
    p    = np.zeros_like(g)
    qx   = x0
    qy   = x0
    u    = x0
    ubar = x0

    error = np.zeros(Niter)
    for k in range(Niter):
        p  = (p + sigma*(A(ubar) - g))/(1+sigma)
        [ubarx, ubary] = my_grad(np.reshape(ubar, [m,n]))
        # print(np.max(np.abs(qx + grad_scale*sigma*ubarx)))
        qx = alpha*(qx + grad_scale*sigma*ubarx)/np.maximum(alpha, np.abs(qx + grad_scale*sigma*ubarx)) 
        qy = alpha*(qy + grad_scale*sigma*ubary)/np.maximum(alpha, np.abs(qy + grad_scale*sigma*ubary))
        
        uiter = np.maximum(0, u - tau*(AT(p) - grad_scale*my_div(qx, qy)))
    
        ubar = uiter + theta*(uiter - u)
        u = ubar
        
        if f is not None:
            error[k] = np.sum(abs(ubar - f)**2)/np.sum(abs(f)**2)
        if print_flag:
            print('TV Iteration: ' + str(k+1) + '/' + str(Niter) + ', Error:' + str(error[k]))
      
    rec = np.reshape(u, [m, n])
    return rec

def ntc(x):
    return torch.Tensor(x).to(config.device).double()

def ctn(x):
    return x.cpu().numpy()

# Load example data
sample = "01a"
ct_data = scipy.io.loadmat(f'data_htc2022/htc2022_test_data/htc2022_{sample}_limited.mat')
ct_data = ct_data["CtDataLimited"][0][0]
sino = ct_data["sinogram"]
angles = ct_data["parameters"]["angles"][0, 0][0]

# Aop = RadonOperator(angles)
# A = lambda x: Aop.forward(x)
# AT = lambda y: Aop.backward(y)
# Amat = get_real_matrix(angles)
# Amat = torch.from_numpy(Amat).to_sparse()
# torch.save(Amat, "A_matrix.pt")
# Amat = Amat.to("cuda")
Amat = torch.load('A_matrix.pt', map_location=torch.device('cuda'))

# A = lambda x: ctn(torch.matmul(Amat, ntc(x).reshape(-1, 1)).reshape(len(angles), 560))
# AT = lambda y: ctn(torch.matmul(Amat.T, ntc(y).reshape(-1, 1)).reshape(512, 512))

# A = lambda x: torch.matmul(Amat, x.reshape(-1, 1)).reshape(len(angles), 560)
# AT = lambda y: torch.matmul(Amat.T, y.reshape(-1, 1)).reshape(512, 512)

A = Amat

sino *= 200

# syn_image = np.load("data_htc2022_simulated/images_without_blur.npy")[1].astype(np.float64)
syn_image = np.load("data_htc2022_simulated/images.npy")[1].astype(np.float32)
# syn_image = np.zeros([512, 512])
# syn_image[128:256,128:256]= 1
syn_image /= np.max(syn_image)
# syn_image *= 0.006

syn_image = torch.Tensor(syn_image).to(config.device).double()
# sino_syn = A(syn_image)
sino_syn = torch.matmul(A, syn_image.reshape(-1, 1)).reshape(181, 560).double()
# sino_syn += 0.003*np.abs(np.max(sino_syn))*np.random.randn(*sino_syn.shape) 

eta   = torch.abs(sino_syn).max()*torch.randn(*sino_syn.shape).to(config.device)
noise = 0.003*eta
sino_syn = sino_syn + noise

# sino_syn *= 0.00002
sino = sino_syn

sino = torch.Tensor(sino).to(config.device)

L = 250
Niter = 1
x0 = torch.zeros([512, 512]).to(config.device).double()
# for alpha in [0.01, 0.05, 0.1, 0.5]:
# alpha = 0.005
# alpha = 0.01 # no noise
alpha = 5
# alpha = 0.5
# Y = tv(x0, A, AT, sino, alpha, L, Niter, f=syn_image, print_flag=True)
Y = tv_torch(x0, A, sino, alpha, L, Niter, f=syn_image, print_flag=True)
Y =
Y = Y.cpu().numpy()
# print(np.sum(Y))
# Y = AT(sino)
# print(np.sum(sino_syn))
plt.figure()
plt.imshow(Y)
plt.colorbar()
plt.savefig("test_tv.png")

# %%

# from classical_reconstructor import alt_grad_solver

# import numpy as np
# import astra
# import torch
# import os 
# import pandas as pd
# from misc.radon_operator import RadonOperator, get_matrix
# import scipy.io
# import matplotlib.pyplot as plt
# import numpy as np


# # # Load example data
# sample = "02c"
# ct_data = scipy.io.loadmat(f'data_htc2022/htc2022_test_data/htc2022_{sample}_limited.mat')
# ct_data = ct_data["CtDataLimited"][0][0]
# sino = ct_data["sinogram"]
# angles = ct_data["parameters"]["angles"][0, 0][0]

# Aop = RadonOperator(angles)
# alph=100
# bet=0
# lr=0.1
# steps=1000
# rec, _, _, _, _ = alt_grad_solver(Aop, sino, start_angle=angles[0],
#                                                 stop_angle=angles[-1], alph=alph,
#                                                 bet=bet, lr=0.000004, steps=40)

# plt.figure()
# plt.imshow(rec)
# plt.colorbar()
# plt.savefig("test_tv.png")