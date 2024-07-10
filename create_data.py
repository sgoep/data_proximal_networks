import numpy as np
import torch
import h5py
import os

from config import config
from functions.total_variation_regularization import tv
from functions.pytorch_radon import get_operators


if not os.path.exists('data'):
    os.makedirs('data')

if not os.path.exists('data/phantom'):
    os.makedirs('data/phantom')
    
if not os.path.exists('data/fbp'):
    os.makedirs('data/fbp')

if not os.path.exists('data/init_regul'):
    os.makedirs('data/init_regul')



image_size = config.image_size
ran_angles = config.ran_angles
n_angles   = config.n_angles
delta      = config.delta

angles = torch.linspace(-ran_angles, ran_angles*(1-1/n_angles), n_angles)
A, AT  = get_operators(angles=angles, n_angles=len(angles), image_size=image_size, circle=True, filtered=False, device=config.device)
_, FBP = get_operators(angles=angles, n_angles=len(angles), image_size=image_size, circle=True, filtered=True, device=config.device)

forward = lambda x: A(torch.Tensor(x)[None,None,:,:].to(config.device)).squeeze().cpu().numpy()
adjoint = lambda y: AT(torch.Tensor(y)[None,None,:,:].to(config.device)).squeeze().cpu().numpy()

L = 10
Niter = 100
norm2 = 0
Num = 1500
data = (h5py.File('randshepp.mat')['data'][:]).transpose([-1,0,1])
for index in range(Num):
    phantom = data[index,:,:]
    phantom = phantom/np.max(phantom.flatten())
    np.save('data/phantom/phantom_'   + str(index), phantom)
    
    f = torch.Tensor(phantom)[None, None, :, :].to(config.device)
    g = A(f)
    
    f     = f.squeeze().cpu().numpy()
    g     = g.squeeze().cpu().numpy()
    eta   = np.abs(g).max()*np.random.randn(*g.shape)
    noise = delta*eta
    gnoise = g + noise
    
    fbp1 = FBP(torch.Tensor(gnoise)[None,None,:,:].to(config.device)).squeeze().cpu().numpy()
    np.save(f'data/fbp/fbp_{index}', fbp1)

    norm2 += np.linalg.norm(noise)/Num
    
    x0 = np.zeros_like(f)
    F = tv(x0, forward, adjoint, gnoise, config.alpha, L, Niter, phantom)
    np.save(f'data/init_regul/init_regul_{str(index)}', F)
    
    print(f'{index}/{Num}')
    
np.save('data/norm2', norm2)
