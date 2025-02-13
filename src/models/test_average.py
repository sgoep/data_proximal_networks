import argparse

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd
import torch  # type: ignore
from matplotlib.colors import Normalize  # type: ignore
from skimage.metrics import mean_squared_error as mse  # type: ignore
from skimage.metrics import peak_signal_noise_ratio as psnr  # type: ignore
from skimage.metrics import structural_similarity as ssim  # type: ignore

from src.data.data_loader import DataLoader
from src.models.unet import UNet
from src.models.utils import extension_with_zero, get_data_loader
from src.utils.load_config import load_config
from src.utils.parser import parse_nested_list
from src.utils.radon_operator import get_radon_operators
from src.utils.test_utils import method_latex_mapping
from src.visualization.latex_figure import generate_latex_figure_block
from src.visualization.visualization import visualization_with_zoom


def test(example, model_names):
    
    config = load_config(example)
    radon_full = get_radon_operators(example)[0]
    chi = np.zeros([config.num_angles_full, config.det_count])
    chi[config.angles_full <= config.phi_limited] = 1

    # errors = dict.fromkeys(model_names, {"MSE": None, "PSNR": None, "SSIM": None})
    errors = {name: {"MSE": None, "PSNR": None, "SSIM": None} for name in model_names}

    errors["landweber"] = {"MSE": None, "PSNR": None, "SSIM": None}
    errors["tv"] = {"MSE": None, "PSNR": None, "SSIM": None}
    errors["ell1"] = {"MSE": None, "PSNR": None, "SSIM": None}
    
    sort_list = ["landweber", "tv", "ell1"] + model_names
    errors = {key: errors[key] for key in sort_list}
    
    # errors_data = {
    #     "$\\radon_\\Omega (\\signal_{{\\rm REC}}) - \\data^\\delta$": {},
    #     "$\\radon_\\Omega (\\signal_{{\\rm REC}}) - \\radon_\\Omega (\\signal^\\star)$": {},
    #     "$\\radon_{{\\Omega^c}} (\\signal_{{\\rm REC}}) - \\radon_{{\\Omega^c}} (\\signal^\\star)$": {}
    # }
    
    errors_data = {name: {"$\\radon_\\Omega (\\signal_{{\\rm REC}}) - \\data^\\delta$": None, "$\\radon_\\Omega (\\signal_{{\\rm REC}}) - \\radon_\\Omega (\\signal^\\star)$": None, "$\\radon_{{\\Omega^c}} (\\signal_{{\\rm REC}}) - \\radon_{{\\Omega^c}} (\\signal^\\star)$": None} for name in model_names}
    errors_data["landweber"] = {"$\\radon_\\Omega (\\signal_{{\\rm REC}}) - \\data^\\delta$": None, "$\\radon_\\Omega (\\signal_{{\\rm REC}}) - \\radon_\\Omega (\\signal^\\star)$": None, "$\\radon_{{\\Omega^c}} (\\signal_{{\\rm REC}}) - \\radon_{{\\Omega^c}} (\\signal^\\star)$": None}
    errors_data["tv"] = {"$\\radon_\\Omega (\\signal_{{\\rm REC}}) - \\data^\\delta$": None, "$\\radon_\\Omega (\\signal_{{\\rm REC}}) - \\radon_\\Omega (\\signal^\\star)$": None, "$\\radon_{{\\Omega^c}} (\\signal_{{\\rm REC}}) - \\radon_{{\\Omega^c}} (\\signal^\\star)$": None}
    errors_data["ell1"] = {"$\\radon_\\Omega (\\signal_{{\\rm REC}}) - \\data^\\delta$": None, "$\\radon_\\Omega (\\signal_{{\\rm REC}}) - \\radon_\\Omega (\\signal^\\star)$": None, "$\\radon_{{\\Omega^c}} (\\signal_{{\\rm REC}}) - \\radon_{{\\Omega^c}} (\\signal^\\star)$": None}
    
    sort_list = ["landweber", "tv", "ell1"] + model_names
    errors_data = {key: errors_data[key] for key in sort_list}

    for initial_regul in ["landweber", "tv", "ell1"]:
        print(f"Initial recon: {initial_regul}")
        initrecon = initial_regul in ["tv", "ell1"]
        model_test_set = DataLoader(id=[i for i in range(200)], initrecon=initrecon, test_train="test", example=example, initial_regul=initial_regul)
        TestDataGen = torch.utils.data.DataLoader(model_test_set, batch_size=1, shuffle=False, num_workers=2)

        mse_list = []
        ssim_list = []
        psnr_list = []
        
        mse_list_data1 = []
        mse_list_data2 = []
        mse_list_data3 = []
        for X, Y, Z in TestDataGen:
            
            radon_X = radon_full.forward(torch.Tensor(X).float().cuda()).cpu().numpy().squeeze()
            radon_Y = radon_full.forward(torch.Tensor(Y).float().cuda()).cpu().numpy().squeeze()
            Z = extension_with_zero(Z, config).cpu().numpy().squeeze()

            mse_list_data1.append(mse(radon_Y.astype('float64') * chi, Z))
            mse_list_data2.append(mse(radon_Y.astype('float64') * chi, radon_X * chi))
            mse_list_data3.append(mse(radon_Y.astype('float64') * (1 - chi), radon_X * (1 - chi)))
            
            # radon_Y = radon_full.forward(X).numpy().squeeze() * chi
            
            X, Y, Z = X.numpy().squeeze(), Y.numpy().squeeze(), Z.squeeze()
            mse_list.append(mse(X, Y))
            psnr_list.append(psnr(X, Y, data_range=Y.max() - Y.min()))
            ssim_list.append(ssim(X, Y, data_range=Y.max() - Y.min()))
        
        mean_mse = np.mean(mse_list)
        mean_ssim = np.mean(ssim_list)
        mean_psnr = np.mean(psnr_list)
        
        print(f'{initial_regul} Standard deviaton mse: {round(np.std(mse_list), 4)}')
        print(f'{initial_regul} Standard deviaton psnr: {round(np.std(psnr_list), 4)}')
        print(f'{initial_regul} Standard deviaton ssim: {round(np.std(ssim_list), 4)}')
        
        errors[initial_regul]["MSE"] = mean_mse
        errors[initial_regul]["PSNR"] = mean_psnr
        errors[initial_regul]["SSIM"] = mean_ssim
        
        mean_mse_data1 = np.mean(mse_list_data1)
        mean_mse_data2 = np.mean(mse_list_data2)
        mean_mse_data3 = np.mean(mse_list_data3)
        
        errors_data[initial_regul]["$\\radon_\\Omega (\\signal_{{\\rm REC}}) - \\data^\\delta$"] = mean_mse_data1
        errors_data[initial_regul]["$\\radon_\\Omega (\\signal_{{\\rm REC}}) - \\radon_\\Omega (\\signal^\\star)$"] = mean_mse_data2
        errors_data[initial_regul]["$\\radon_{{\\Omega^c}} (\\signal_{{\\rm REC}}) - \\radon_{{\\Omega^c}} (\\signal^\\star)$"] = mean_mse_data3
        
        print(mean_mse)
        print(mean_psnr)
        print(mean_ssim)
        
        print(mean_mse_data1)
        print(mean_mse_data2)
        print(mean_mse_data3)
        # print(mean_mse_data)

    for model_name in model_names:
        print(f"Model: {model_name}.")
        mse_list = []
        ssim_list = []
        psnr_list = []
        
        mse_list_data1 = []
        mse_list_data2 = []
        mse_list_data3 = []

        initial_regul = model_name.split("_")[0]
        initrecon = initial_regul in ["tv", "ell1"]
    
        model = UNet(1, 1, model_name, example).to("cuda")
        model.load_state_dict(torch.load(f"models/{example}/{model_name}.pth"))
        
        model_test_set = DataLoader(id=[i for i in range(200)], initrecon=initrecon, test_train="test", example=example, initial_regul=initial_regul,)
        TestDataGen = torch.utils.data.DataLoader(model_test_set, batch_size=1, shuffle= False, num_workers=2)
        
        for X, Y, Z in TestDataGen:
            X = X.numpy().squeeze()
            Y = torch.Tensor(Y).to(device="cuda", dtype=torch.float)
            model_output = model(Y).detach()
            
            radon_X = radon_full.forward(torch.Tensor(X).cuda()).cpu().numpy().squeeze()
            radon_Y = radon_full.forward(torch.Tensor(Y).cuda()).cpu().numpy().squeeze()
            Z = extension_with_zero(Z, config).cpu().numpy().squeeze()

            radon_output = radon_full.forward(model_output).cpu().numpy().squeeze()

            mse_list_data1.append(mse(radon_output.astype('float64') * chi, Z))
            mse_list_data2.append(mse(radon_output.astype('float64') * chi, radon_X * chi))
            mse_list_data3.append(mse(radon_output.astype('float64') * (1 - chi), radon_X * (1 - chi)))
            
            model_output = model_output.cpu().squeeze().numpy()
            
            mse_list.append(mse(X, model_output))
            psnr_list.append(psnr(X, model_output, data_range=model_output.max() - model_output.min()))
            ssim_list.append(ssim(X, model_output, data_range=model_output.max() - model_output.min()))
            
        mean_mse = np.mean(mse_list)
        mean_ssim = np.mean(ssim_list)
        mean_psnr = np.mean(psnr_list)
        
        errors[model_name]["MSE"] = mean_mse
        errors[model_name]["PSNR"] = mean_psnr
        errors[model_name]["SSIM"] = mean_ssim
        
        print(f'{model_name} Standard deviaton mse: {round(np.std(mse_list), 4)}')
        print(f'{model_name} Standard deviaton psnr: {round(np.std(psnr_list), 4)}')
        print(f'{model_name} Standard deviaton ssim: {round(np.std(ssim_list), 4)}')
        
        mean_mse_data1 = np.mean(mse_list_data1)
        mean_mse_data2 = np.mean(mse_list_data2)
        mean_mse_data3 = np.mean(mse_list_data3)
        
        errors_data[model_name]["$\\radon_\\Omega (\\signal_{{\\rm REC}}) - \\data^\\delta$"] = mean_mse_data1
        errors_data[model_name]["$\\radon_\\Omega (\\signal_{{\\rm REC}}) - \\radon_\\Omega (\\signal^\\star)$"] = mean_mse_data2
        errors_data[model_name]["$\\radon_{{\\Omega^c}} (\\signal_{{\\rm REC}}) - \\radon_{{\\Omega^c}} (\\signal^\\star)$"] = mean_mse_data3

        print(mean_mse)
        print(mean_psnr)
        print(mean_ssim)
        
        print(mean_mse_data1)
        print(mean_mse_data2)
        print(mean_mse_data3)
    
    df = pd.DataFrame.from_dict(errors)
    df.rename(columns=method_latex_mapping, inplace=True)
    df = df.T
    latex_output = df.to_latex(
            index=True,
            formatters={"name": str.upper},
            float_format="{:.4f}".format,
        )
    print(latex_output)
    
    df_data = pd.DataFrame.from_dict(errors_data)
    df_data.rename(columns=method_latex_mapping, inplace=True)
    df_data = df_data.T
    latex_output_data = df_data.to_latex(
            index=True,
            formatters={"name": str.upper},
            float_format="{:.4f}".format,
        )
    print(latex_output_data)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing inputs.")
    parser.add_argument("example", type=str, help="Which example.")
    parser.add_argument("model_names", type=parse_nested_list, nargs='?')

    args = parser.parse_args()
    if args.model_names is None:
        from example import models
        model_names = models
    print(f"Testing for example {args.example}")
    print(f"Models: {model_names}")
    print("##############################################################")
    test(example=args.example, model_names=model_names)
    print(f"Testing completed for {args.example}.")
