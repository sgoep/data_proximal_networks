
import numpy as np
import torch
import h5py
import os

from config import config
from total_variation_regularization_astra import tv
from misc.radon_operator import ram_lak_filter, get_matrix

if not os.path.exists('data_astra'):
    os.makedirs('data_astra')

if not os.path.exists('data_astra/phantom'):
    os.makedirs('data_astra/phantom')
    
if not os.path.exists('data_astra/fbp'):
    os.makedirs('data_astra/fbp')

if not os.path.exists('data_astra/init_regul'):
    os.makedirs('data_astra/init_regul')



# image_size = config.image_size
# ran_angles = config.ran_angles
# n_angles   = config.n_angles
# delta      = config.delta

image_size = 128
delta = 0.05

Ns = 200
Nal = 180
Phi = np.pi/3
al_full = np.linspace(-np.pi/2, np.pi/2*(1-1/Nal), Nal, endpoint=True)
al1 = al_full[abs(al_full)<=Phi]
# al2 = al_full[np.abs(al_full)>Phi]

A = get_matrix(image_size, Ns, al1)
A = A.to(config.device)

alpha = 0.1
L = 200
Niter = 500
norm2 = 0
Num = 1500
data = (h5py.File('randshepp.mat')['data'][:]).transpose([-1,0,1])
x0 = torch.Tensor(np.zeros([image_size, image_size])).to(config.device)
for index in range(Num):
    phantom = data[index,:,:]
    phantom = phantom/np.max(phantom.flatten())
    np.save('data_astra_delta005/phantom/phantom_'   + str(index), phantom)
    
    f = torch.Tensor(phantom).to(config.device)
    g = A.matmul(f.reshape(-1, 1)).reshape(len(al1), Ns)

    eta   = torch.abs(g).max()*torch.randn(*g.shape).to(config.device)
    noise = delta*eta
    gnoise = g + noise
    
    gfilt = ram_lak_filter(gnoise)
    gfilt = torch.Tensor(gfilt).to(config.device)
    # fbp1 = FBP(torch.Tensor(gnoise).to(config.device)).squeeze().cpu().numpy()
    fbp1 = A.T.matmul(gfilt.reshape(-1, 1)).reshape(image_size, image_size)
    np.save(f'data_astra_delta005/fbp/fbp_{index}', fbp1.squeeze().cpu().numpy())

    norm2 += torch.linalg.norm(noise)/Num
    
    gnoise = torch.Tensor(gnoise).to(config.device)
    F = tv(x0, A, gnoise, alpha, L, Niter, torch.Tensor(phantom).to(config.device), print=False)
    np.save(f'data_astra_delta005/init_regul/init_regul_{str(index)}', F.squeeze().cpu().numpy())
    
    print(f'{index}/{Num}, device: {config.device}')
    
np.save('data_astra_delta005/norm2', norm2.cpu().numpy())

# %%
# import numpy as np
# import matplotlib.pyplot as plt
# x = np.load('./data_astra/init_regul/init_regul_19.npy')
# plt.imshow(x)
# plt.colorbar()