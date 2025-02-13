
import numpy as np
import torch
import h5py
import os
import scipy
from config import config
from total_variation_regularization_astra_real import tv
from misc.radon_operator import RadonOperator

# Load example data
sample = "01a"
ct_data = scipy.io.loadmat(f'data_htc2022/htc2022_test_data/htc2022_{sample}_limited.mat')
ct_data = ct_data["CtDataLimited"][0][0]
sino = ct_data["sinogram"]
angles = ct_data["parameters"]["angles"][0, 0][0]

Aop = RadonOperator(angles)
A = lambda x: Aop.forward(x)
AT = lambda y: Aop.backward(y)
# Afbp = lambda z: Aop.fbp(z)

L = 300
Niter = 1000
x0 = np.zeros([512, 512])
alpha = 0.1
Num = 1500
norm2 = 0
images = np.load("data_htc2022_simulated/images.npy").astype(np.float64)
phantom_all = np.zeros([Num, 512, 512])
fbp_all = np.zeros([Num, 512, 512])
init_regul_all = np.zeros([Num, 512, 512])

for index in range(Num):
    np.random.seed(index)
    phantom = images[index,:,:]
    phantom = phantom/np.max(phantom)
    phantom_all[index, :, :] = phantom
    
    sino = Aop.forward(phantom)
    noise = 0.003*np.abs(np.max(sino))*np.random.randn(*sino.shape) 
    sino += noise
    
    fbp = Aop.fbp(sino)
    # np.save(f'data_htc2022_simulated/fbp/fbp_{index}', fbp)
    fbp_all[index, :, :] = fbp

    norm2 += np.linalg.norm(noise)/Num
    
    F = tv(x0, A, AT, sino, alpha, L, Niter, print_flag=False)
    # np.save(f'data_htc2022_simulated/init_regul/init_regul_{str(index)}', F)
    init_regul_all[index, :, :] = F
    
    print(f'{index+1}/{Num}')
    
np.save('data_htc2022_simulated/norm2', norm2)
np.save('data_htc2022_simulated/phantom', phantom_all)
np.save('data_htc2022_simulated/fbp', fbp_all)
np.save('data_htc2022_simulated/init_regul', init_regul_all)

# %%

# %%
# import numpy as np
# import matplotlib.pyplot as plt
# x = np.load('./data_astra/init_regul/init_regul_19.npy')
# plt.imshow(x)
# plt.colorbar()