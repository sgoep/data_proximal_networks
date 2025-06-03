import numpy as np
import pandas as pd
from skimage.metrics import mean_squared_error as mse  # type: ignore
from skimage.metrics import peak_signal_noise_ratio as psnr  # type: ignore
from skimage.metrics import structural_similarity as ssim  # type: ignore

metrics = ["MSE", "PSNR", "SSIM"]
method_latex_mapping = {
    "fbp": "$\\signal_{\\rm FBP}$",
    "landweber": "$\\signal_{\\rm LANDWEBER}$",
    "tv": "$\\signal_{\\rm TV}$",
    "ell1": "$\\signal_{\\ell_1}$",
    ####################################################
    "fbp_X_res": "$\\signal_{\\rm FBP}^{\\rm RES}$",
    "landweber_X_res": "$\\signal_{\\rm LANDWEBER}^{\\rm RES}$",
    "tv_X_res": "$\\signal_{\\rm TV}^{\\rm RES}$",
    "ell1_X_res": "$\\signal_{\\ell_1}^{\\rm RES}$",
    ####################################################
    "fbp_nsn": "$\\signal_{\\rm FBP}^{\\rm NSN}$",
    "landweber_nsn": "$\\signal_{\\rm LANDWEBER}^{\\rm NSN}$",
    "tv_nsn": "$\\signal_{\\rm TV}^{\\rm NSN}$",
    "ell1_nsn": "$\\signal_{\\ell_1}^{\\rm NSN}$",
    ####################################################
    "fbp_single_dp": "$\\signal_{\\rm FBP}^{\\rm DP}$",
    "landweber_single_dp": "$\\signal_{\\rm LANDWEBER}^{\\rm DP}$",
    "tv_single_dp": "$\\signal_{\\rm TV}^{\\rm DP}$",
    "ell1_single_dp": "$\\signal_{\\ell_1}^{\\rm DP}$",
    ####################################################
    ####################################################
    "fbp_smooth_nsn": "$\\signal_{\\rm FBP}^{\\rm NSN}$ smooth",
    "landweber_smooth_nsn": "$\\signal_{\\rm LANDWEBER}^{\\rm NSN}$ smooth",
    "tv_smooth_nsn": "$\\signal_{\\rm TV}^{\\rm NSN}$ smooth",
    "ell1_smooth_nsn": "$\\signal_{\\ell_1}^{\\rm NSN}$ smooth",
    ####################################################
    "fbp_smooth_single_dp": "$\\signal_{\\rm FBP}^{\\rm DP}$ smooth",
    "landweber_smooth_single_dp": "$\\signal_{\\rm LANDWEBER}^{\\rm DP}$ smooth",
    "tv_smooth_single_dp": "$\\signal_{\\rm TV}^{\\rm DP}$ smooth",
    "ell1_smooth_single_dp": "$\\signal_{\\ell_1}^{\\rm DP}$ smooth",
    ####################################################
    ############# DATA
    ####################################################
    "fbp_data": "$\\forward(\\signal_{\\rm FBP}) - \\forward(\\signal^\\star)$",
    "landweber_data": "$\\forward(\\signal_{\\rm LANDWEBER}) - \\forward(\\signal^\\star)$",
    "tv_data": "$\\forward(\\signal_{\\rm TV}) - \\forward(\\signal^\\star)$",
    "ell1_data": "$\\forward(\\signal_{\\ell_1}) - \\forward(\\signal^\\star)$",
    ####################################################
    "fbp_X_res_data": "$\\forward(\\signal_{\\rm FBP}^{\\rm RES}$) - \\forward(\\signal^\\star)$",
    "landweber_X_res_data": "$\\forward(\\signal_{\\rm LANDWEBER}^{\\rm RES}) - \\forward(\\signal^\\star)$",
    "tv_X_res_data": "$\\forward(\\signal_{\\rm TV}^{\\rm RES}) - \\forward(\\signal^\\star)$",
    "ell1_X_res_data": "$\\forward(\\signal_{\\ell_1}^{\\rm RES}) - \\forward(\\signal^\\star)$",
    ####################################################
    "fbp_nsn_data": "$\\forward(\\signal_{\\rm FBP}^{\\rm NSN}$) - \\forward(\\signal^\\star)$",
    "landweber_nsn_data": "$\\forward(\\signal_{\\rm LANDWEBER}^{\\rm NSN}) - \\forward(\\signal^\\star)$",
    "tv_nsn_data": "$\\forward(\\signal_{\\rm TV}^{\\rm NSN}) - \\forward(\\signal^\\star)$",
    "ell1_nsn_data": "$\\forward(\\signal_{\\ell_1}^{\\rm NSN}) - \\forward(\\signal^\\star)$",
    ####################################################
    "fbp_single_dp_data": "$\\forward(\\signal_{\\rm FBP}^{\\rm DP}$) - \\forward(\\signal^\\star)$",
    "landweber_single_dp_data": "$\\forward(\\signal_{\\rm LANDWEBER}^{\\rm DP}) - \\forward(\\signal^\\star)$",
    "tv_single_dp_data": "$\\forward(\\signal_{\\rm TV}^{\\rm DP}) - \\forward(\\signal^\\star)$",
    "ell1_single_dp_data": "$\\forward(\\signal_{\\ell_1}^{\\rm DP}) - \\forward(\\signal^\\star)$",
}


def calc_errors(recon, X):
    return {
        "MSE": mse(recon, X),
        "PSNR": psnr(X, recon, data_range=recon.max() - recon.min()),
        "SSIM": ssim(X, recon, data_range=recon.max() - recon.min()),
    }


def print_errors(recon, X):
    errors = calc_errors(recon, X)
    for key, val in errors.items():
        print(f"{key}: {val}")


def create_error_table(errors, table_path, example, data=False):
    error_table = (
        pd.DataFrame(
            {m: [errors[method][m] for method in errors] for m in metrics},
            index=[method_latex_mapping[k] for k in errors.keys()],
        )
        .reset_index()
        .rename(columns={"index": "methods"})
    )

    error_table.to_csv(f"{table_path}")

    latex_output = error_table.to_latex(
        index=False,
        formatters={"name": str.upper},
        float_format="{:.4f}".format,
    )
    print(latex_output)

    if data:
        with open(f"results/{example}_data_errors_latex.txt", "w") as f:
            f.write(latex_output)
    else:
        with open(f"results/{example}_recon_errors_latex.txt", "w") as f:
            f.write(latex_output)
