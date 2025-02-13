import os
import pandas as pd
import torch
from config import config
from functions.visualization import visualization_with_zoom
from misc.data_loader import DataLoader
from misc.unet_astra import UNet
from skimage.metrics import structural_similarity   as ssim
from skimage.metrics import mean_squared_error      as mse
from skimage.metrics import peak_signal_noise_ratio as psnr

index = 1366

errors = {}

# CONSTRAINT = True
# INITRECON  = True

# Get FBP reconstruction
D = DataLoader(index, False, False)
X, Y = D[index]

errors["fbp"] = {
    "MSE": mse(X.flatten(), Y.flatten()),
    "SSIM": ssim(X.flatten(), Y.flatten(), data_range=X.max() - X.min()),
    "PSNR": psnr(X.flatten(), Y.flatten())
}

# Get TV reconstruction
D = DataLoader(index, False, True)
X, Y = D[index]

errors["tv"] = {
    "MSE": mse(X.flatten(), Y.flatten()),
    "SSIM": ssim(X.flatten(), Y.flatten(), data_range=X.max() - X.min()),
    "PSNR": psnr(X.flatten(), Y.flatten())
}

# for CONSTRAINT in [True, False]:
    # for INITRECON in [True, False]:

        # model_name = f'data_prox_network_constraint_{str(CONSTRAINT)}_init_regul_{str(INITRECON)}'

for model_name in ["FBP_RES", "TV_RES", "FBP_NSN", "TV_NSN", "TV_DP"]:

    if model_name in ["FBP_RES", "FBP_NSN"]:
        CONSTRAINT = False
        INITRECON  = False
    elif model_name in ["TV_RES", "TV_NSN"]:
        CONSTRAINT = False
        INITRECON  = True
    elif model_name == "TV_DP":
        CONSTRAINT = True
        INITRECON  = True
    else:
        pass

    if model_name in ["FBP_RES", "TV_RES"]:
        null_space_network = False
    else:
        null_space_network = True

    D = DataLoader(index, CONSTRAINT, INITRECON)
    X, Y = D[index]

    model = UNet(1, 1, CONSTRAINT, null_space_network)
    model.load_state_dict(torch.load('models_astra_delta005/' + model_name, map_location=torch.device(config.device)))
    model.eval()
    model.to(config.device)

    Y = torch.Tensor(Y).to(config.device)
    # print(Y.is_cuda)
    Y = Y.unsqueeze(0)

    out, res = model(Y)
    out = out.cpu().detach().numpy().squeeze()
    out = out.astype('float32')
    res = res.cpu().detach().numpy().squeeze()
    res = res.astype('float32')

    print(f'MSE:  {mse(X.flatten(), out.flatten())}')
    print(f'SSIM: {ssim(X.flatten(), out.flatten(), data_range=X.max() - X.min())}')
    print(f'PSNR: {psnr(X.flatten(), out.flatten())}')

    errors[model_name] = {
        "MSE": mse(X.flatten(), out.flatten()),
        "SSIM": ssim(X.flatten(), out.flatten(), data_range=X.max() - X.min()),
        "PSNR": psnr(X.flatten(), out.flatten())
    }
        
    visualization_with_zoom(out.T, True, f'results_astra_delta005/{model_name}.pdf')

error_table = pd.DataFrame.from_dict(errors).transpose()
error_table.to_csv("results_astra_delta005/error_table.csv")