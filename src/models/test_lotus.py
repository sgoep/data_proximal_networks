from typing import List

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch_radon
from matplotlib.colors import Normalize
from pytorch_wavelets import DWTForward, DWTInverse

from src.algorithms.ell1_wavelet import ell1_wavelet
from src.algorithms.total_variation import tv
from src.models.unet import UNet
from src.utils.radon_operator import (
    filter_sinogram,
    get_radon_operator,
    get_radon_operators,
)
from src.utils.test_utils import calc_errors, create_error_table, print_errors
from src.visualization.latex_figure import generate_latex_figure_block
from src.visualization.visualization import visualization_with_zoom


def test_other_function_class(which_models, model_names, zoom=False, colorbar=False):

    example = "lotus"

    f = scipy.io.loadmat("data/LotusData256.mat")

    data = f["m"].T

    wavelet = DWTForward(J=5, mode="zero", wave="db3").cuda()
    iwavelet = DWTInverse(mode="zero", wave="db3").cuda()

    angles = np.linspace(0, 360, data.shape[0], endpoint=True)
    angles = angles * np.pi / 180
    N = 256

    source_distance = 2240 * 540 / 120
    origin_detector_distance = 630
    fan_sensor_spacing = 540 / 630
    num_detectors = data.shape[1]

    radon_full = torch_radon.RadonFanbeam(
        N,
        angles,
        source_distance,
        det_count=num_detectors,
        det_spacing=fan_sensor_spacing,
        clip_to_circle=False,
    )

    angles = angles[angles <= 3 * np.pi / 4]

    X = np.zeros([N, N])

    data = torch.Tensor(data).cuda()
    data = data[0 : len(angles), :]

    A = torch_radon.RadonFanbeam(
        N,
        angles,
        source_distance,
        det_count=num_detectors,
        det_spacing=fan_sensor_spacing,
        clip_to_circle=False,
    )

    rec_fbp = A.backward(filter_sinogram(data))

    landweber = torch_radon.solvers.Landweber(A, projection=None, grad=False)

    x0 = torch.zeros([N, N]).cuda()

    rec_landweber = landweber.run(x0, data, 1e-6, iterations=10000)

    L = 500
    Niter = 1000
    alpha = 0.005
    tau = 1 / L
    sigma = 1 / L
    theta = 1
    rec_tv = tv(
        x0,
        A,
        data,
        alpha,
        tau,
        sigma,
        theta,
        L,
        Niter,
        ground_truth=None,
        print_flag=False,
    )

    Niter = 100

    wavelet = DWTForward(J=5, mode="zero", wave="db3").cuda()
    iwavelet = DWTInverse(mode="zero", wave="db3").cuda()

    rec_ell1 = ell1_wavelet(
        wavelet,
        iwavelet,
        A,
        data,
        p_0=0.00009,
        p_1=0.00001,
        Niter=500,
        ground_truth=None,
        print_flag=False,
    )

    # visualization_with_zoom(
    #     example,
    #     X, False, False,
    #     "results/figures/lotus/0000-lotus_ground_truth.pdf"
    # )

    errors = {}
    errors_data = {}
    diff_plots = {}

    figure_names_diff = []
    captions_diff = []
    labels_diff = []

    # Process FBP# Print the errors
    print("fbp.")
    errors["fbp"] = calc_errors(rec_fbp.cpu().numpy().squeeze().astype("float64"), X)
    print_errors(rec_fbp.cpu().numpy().squeeze().astype("float64"), X)

    # Visualize the result
    visualization_with_zoom(
        example,
        rec_fbp.cpu().numpy().squeeze().astype("float64"),
        zoom,
        colorbar,
        "results/figures/lotus/0000-lotus_fbp.pdf",
    )
    print("##############################################################")

    print("landweber.")
    errors["landweber"] = calc_errors(
        rec_landweber.cpu().numpy().squeeze().astype("float64"), X
    )
    print_errors(rec_landweber.cpu().numpy().squeeze().astype("float64"), X)
    # Visualize the result
    visualization_with_zoom(
        example,
        rec_landweber.cpu().numpy().squeeze().astype("float64"),
        zoom,
        colorbar,
        "results/figures/lotus/0000-lotus_landweber.pdf",
    )
    print("##############################################################")

    print("tv.")
    # Process TV
    errors["tv"] = calc_errors(rec_tv.cpu().numpy().squeeze().astype("float64"), X)
    print_errors(rec_tv.cpu().numpy().squeeze().astype("float64"), X)

    # Visualize the result
    visualization_with_zoom(
        example,
        rec_tv.cpu().numpy().squeeze().astype("float64"),
        zoom,
        colorbar,
        "results/figures/lotus/0000-lotus_tv.pdf",
    )
    print("##############################################################")

    # Process ELL1
    print("ell1.")
    errors["ell1"] = calc_errors(rec_ell1.cpu().numpy().squeeze().astype("float64"), X)
    print_errors(rec_ell1.cpu().numpy().squeeze().astype("float64"), X)

    # Visualize the result
    visualization_with_zoom(
        example,
        rec_ell1.cpu().numpy().squeeze().astype("float64"),
        zoom,
        colorbar,
        "results/figures/lotus/0000-lotus_ell1.pdf",
    )
    print("##############################################################")

    print(f"Models: {model_names}")
    for model_name in model_names:

        if model_name.split("_")[0] == "tv":
            Y = rec_tv
        elif model_name.split("_")[0] == "ell1":
            Y = rec_ell1
        elif model_name.split("_")[0] == "landweber":
            Y = rec_landweber
        else:
            Y = rec_fbp
        Z = []

        # X = torch.Tensor(X).cuda()
        Y = torch.Tensor(Y).cuda()
        # Z = torch.Tensor(Z).to(config.device)

        Y = Y.squeeze().unsqueeze(0).unsqueeze(0)
        # Z = Z.unsqueeze(0).unsqueeze(0)

        print(f"Model: {model_name}.")

        # radon_full, radon_limited, radon_null_space = get_radon_operators(
        #     example
        # )
        model = UNet(1, 1, model_name, example, radon_full=radon_full)
        model.load_state_dict(torch.load(f"models/{which_models}/{model_name}.pth"))
        model.cuda()

        # diff_recons = abs(radon_limited.forward(model_output) - radon_limited.forward(Y))#model_output - Y
        # radon_diff_recons = diff_recons  # radon_full.forward(diff_recons)
        # diff_plots[f"results/figures/lotus/0000-lotus_diff_{model_name}.pdf"] = abs(
        # radon_diff_recons.cpu().detach().numpy().squeeze().astype('float64'))
        # errors_data[f"{model_name}_data"] = calc_errors(
        # radon_limited.forward(model_output).cpu().detach().numpy().squeeze().astype('float64'),
        # radon_limited.forward(Y).cpu().numpy().squeeze().astype('float64'),
        # )
        # mse_error_diff = errors_data[f"{model_name}_data"]["MSE"]
        # figure_names_diff.append(f"0000-lotus_diff_{model_name}.pdf")
        # captions_diff.append(
        # f"|\\forward(\\signal_{{\\rm {model_name.split('_')[0].upper()}}}^{{{model_name.split('_')[-1].upper()}}}) - \\forward(\\signal_{{\\rm {model_name.split('_')[0].upper()}}})|, \\text{{MSE}}: {round(mse_error_diff, 4)}"
        # )
        # labels_diff.append(f"0000-lotus_diff_{model_name}.pdf")

        output = model(Y).cpu().detach().numpy().squeeze()

        print_errors(output, X)

        print("Plotting.")
        visualization_with_zoom(
            example,
            output,
            zoom,
            colorbar,
            f"results/figures/lotus/0000-lotus_{model_name}.pdf",
        )
        print(f"Finished {model_name}.")

        errors[model_name] = calc_errors(output, X)
        print("##############################################################")

    table_path = "results/lotus_error_table.csv"
    create_error_table(errors, table_path, example)

    # Normalize and save error plots
    norm = Normalize(
        vmin=min(v.min() for v in diff_plots.values()),
        vmax=max(v.max() for v in diff_plots.values()),
    )
    for path, val in diff_plots.items():
        plt.figure()
        plt.imshow(val, norm=norm, cmap="gray")
        plt.axis("off")
        plt.colorbar()
        plt.savefig(path, bbox_inches="tight")

    # print(generate_latex_figure_block(figure_names_diff, captions_diff, labels_diff))
    # latex_figure_diff = generate_latex_figure_block(figure_names_diff, captions_diff, labels_diff)
    # with open("results/lotus_latex_diff_figure.txt", 'w') as f:
    # f.write(latex_figure_diff)

    # create_error_table(
    # errors, f"results/lotus_recon_error_table.csv", "lotus", False)

    print("Finished.")


if __name__ == "__main__":
    import argparse

    from src.utils.parser import parse_nested_list

    parser = argparse.ArgumentParser(description="Testing inputs.")
    parser.add_argument("which_models", type=str, help="Which models.")
    parser.add_argument("model_names", type=parse_nested_list, nargs="?")

    # Parse arguments from the command line
    args = parser.parse_args()

    if args.model_names is None:
        from example import models

        model_names = models

    print(f"Testing for models trained with {args.which_models}")
    print(f"Models: {model_names}")
    print("##############################################################")
    test_other_function_class(which_models=args.which_models, model_names=model_names)
    print(f"Testing completed for {args.which_models}.")
