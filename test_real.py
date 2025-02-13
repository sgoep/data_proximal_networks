# %%
import os
import pandas as pd
import torch
from config import config
from functions.visualization import visualization_with_zoom
from misc.data_loader_real import DataLoader
from misc.unet_real import UNet
from skimage.metrics import structural_similarity   as ssim
from skimage.metrics import mean_squared_error      as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
import scipy
from misc.radon_operator import RadonOperator
import matplotlib.pyplot as plt

errors = {}

# CONSTRAINT = True
# INITRECON  = True

# Load example data
sample = "01a"
ct_data = scipy.io.loadmat(f'data_htc2022/htc2022_test_data/htc2022_{sample}_limited.mat')
ct_data = ct_data["CtDataLimited"][0][0]
sino = ct_data["sinogram"]
angles = ct_data["parameters"]["angles"][0, 0][0]

sino *= 200

Aop = RadonOperator(angles)
Y_fbp = Aop.fbp(sino)
plt.figure()
plt.imshow(Y_fbp)
plt.colorbar()
plt.savefig('results_real/fbp_limited.pdf')


ct_data_full = scipy.io.loadmat(f'data_htc2022/htc2022_test_data/htc2022_{sample}_full.mat')
ct_data_full = ct_data_full["CtDataFull"][0][0]
sino_full = ct_data_full["sinogram"]
angles_full = ct_data_full["parameters"]["angles"][0, 0][0]
Aop_full = RadonOperator(angles_full)

sino_full *= 200

X = Aop_full.fbp(sino_full)

plt.figure()
plt.imshow(X.T)
plt.colorbar()
plt.savefig('results_real/fbp_full.pdf')
# visualization_with_zoom(X.T, True, f'results_real/fbp_full.pdf')



errors["fbp"] = {
    "MSE": mse(X.flatten(), Y_fbp.flatten()),
    "SSIM": ssim(X.flatten(), Y_fbp.flatten(), data_range=X.max() - X.min()),
    "PSNR": psnr(X.flatten(), Y_fbp.flatten(), data_range=X.max() - X.min())
}


# Get TV reconstruction
import numpy as np
from total_variation_regularization_astra_real import tv

L = 300
Niter = 1000
x0 = np.zeros([512, 512])
alpha = 0.5
A = lambda x: Aop.forward(x)
AT = lambda y: Aop.backward(y)
Y_tv = tv(x0, A, AT, sino, alpha, L, Niter, print_flag=False)
plt.figure()
plt.imshow(Y_tv.T)
plt.colorbar()
plt.savefig('results_real/tv.pdf')

errors["tv"] = {
    "MSE": mse(X.flatten(), Y_tv.flatten()),
    "SSIM": ssim(X.flatten(), Y_tv.flatten(), data_range=X.max() - X.min()),
    "PSNR": psnr(X.flatten(), Y_tv.flatten(), data_range=X.max() - X.min())
}

# for CONSTRAINT in [True, False]:
    # for INITRECON in [True, False]:

        # model_name = f'data_prox_network_constraint_{str(CONSTRAINT)}_init_regul_{str(INITRECON)}'

for model_name in ["FBP_RES", "TV_RES", "FBP_NSN", "TV_NSN", "TV_DP"]:

    if model_name in ["FBP_RES", "FBP_NSN"]:
        CONSTRAINT = False
        Y = Y_fbp
    elif model_name in ["TV_RES", "TV_NSN"]:
        CONSTRAINT = False
        Y = Y_tv
    elif model_name == "TV_DP":
        CONSTRAINT = True
        Y = Y_tv
    else:
        pass

    if model_name in ["FBP_RES", "TV_RES"]:
        null_space_network = False
    else:
        null_space_network = True

    with torch.no_grad():
        model = UNet(1, 1, CONSTRAINT, null_space_network)
        model.load_state_dict(torch.load('models_real/' + model_name, map_location=torch.device(config.device)))
        model.eval()
        model.to(config.device)

        Y = torch.Tensor(Y).to(config.device)
        # print(Y.is_cuda)
        Y = Y.unsqueeze(0).unsqueeze(0)

        print(Y.shape)
        out, res = model(Y)
        out = out.cpu().detach().numpy().squeeze()
        out = out.astype('float32')
        res = res.cpu().detach().numpy().squeeze()
        res = res.astype('float32')

        print(f'MSE:  {mse(X.flatten(), out.flatten())}')
        print(f'SSIM: {ssim(X.flatten(), out.flatten(), data_range=X.max() - X.min())}')
        print(f'PSNR: {psnr(X.flatten(), out.flatten(), data_range=X.max() - X.min())}')

        errors[model_name] = {
            "MSE": mse(X.flatten(), out.flatten()),
            "SSIM": ssim(X.flatten(), out.flatten(), data_range=X.max() - X.min()),
            "PSNR": psnr(X.flatten(), out.flatten(), data_range=X.max() - X.min())
        }
            
        visualization_with_zoom(out.T, True, f'results_real/{model_name}.pdf')

error_table = pd.DataFrame.from_dict(errors).transpose()
error_table.to_csv("results_real/error_table.csv")