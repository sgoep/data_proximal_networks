from typing import List

import matplotlib.pyplot as plt
import numpy as np  # type: ignore
import torch  # type: ignore
import torch_radon
from matplotlib.colors import Normalize  # type: ignore

from src.data.data_loader import DataLoader
from src.data.utils import create_ellipse_example, create_lotus_example
from src.models.unet import UNet
from src.utils.load_config import load_config
from src.utils.radon_operator import (
    filter_sinogram,
    get_radon_operator,
    get_radon_operators,
)
from src.utils.test_utils import calc_errors, create_error_table, print_errors
from src.visualization.latex_figure import generate_latex_figure_block
from src.visualization.visualization import visualization_with_zoom


def test_other_function_class(example, model_names, zoom=False, colorbar=False):
    print(example)
    (X, data_limited, fbp_ellipse, tv_ellipse, ell1_ellipse, landweber_ellipse) = (
        create_ellipse_example(example)
    )
    # (X, data_limited, fbp_ellipse,
    #  tv_ellipse, ell1_ellipse, landweber_ellipse) = create_lotus_example()

    visualization_with_zoom(
        example,
        X.cpu().numpy().squeeze().astype("float64"),
        False,
        False,
        "results/figures/ellipse/0000-ellipse_ground_truth.pdf",
    )

    config = load_config(example)

    errors = {}
    errors_data = {}
    diff_plots = {}

    figure_names_diff = []
    captions_diff = []
    labels_diff = []

    if True:
        # Process FBP# Print the errors
        print("fbp.")
        errors["fbp"] = calc_errors(
            fbp_ellipse.cpu().numpy().squeeze().astype("float64"),
            X.cpu().numpy().squeeze().astype("float64"),
        )
        print_errors(
            fbp_ellipse.cpu().numpy().squeeze().astype("float64"),
            X.cpu().numpy().squeeze().astype("float64"),
        )

        # Visualize the result
        visualization_with_zoom(
            example,
            fbp_ellipse.cpu().numpy().squeeze().astype("float64"),
            zoom,
            colorbar,
            "results/figures/ellipse/0000-ellipse_fbp.pdf",
        )
        print("##############################################################")

        print("landweber.")
        errors["landweber"] = calc_errors(
            landweber_ellipse.cpu().numpy().squeeze().astype("float64"),
            X.cpu().numpy().squeeze().astype("float64"),
        )
        print_errors(
            landweber_ellipse.cpu().numpy().squeeze().astype("float64"),
            X.cpu().numpy().squeeze().astype("float64"),
        )
        # Visualize the result
        visualization_with_zoom(
            example,
            landweber_ellipse.cpu().numpy().squeeze().astype("float64"),
            zoom,
            colorbar,
            "results/figures/ellipse/0000-ellipse_landweber.pdf",
        )
        print("##############################################################")

        print("tv.")
        # Process TV
        errors["tv"] = calc_errors(
            tv_ellipse.cpu().numpy().squeeze().astype("float64"),
            X.cpu().numpy().squeeze().astype("float64"),
        )
        print_errors(
            tv_ellipse.cpu().numpy().squeeze().astype("float64"),
            X.cpu().numpy().squeeze().astype("float64"),
        )

        # Visualize the result
        visualization_with_zoom(
            example,
            tv_ellipse.cpu().numpy().squeeze().astype("float64"),
            zoom,
            colorbar,
            "results/figures/ellipse/0000-ellipse_tv.pdf",
        )
        print("##############################################################")

        # Process ELL1
        print("ell1.")
        errors["ell1"] = calc_errors(
            ell1_ellipse.cpu().numpy().squeeze().astype("float64"),
            X.cpu().numpy().squeeze().astype("float64"),
        )
        print_errors(
            ell1_ellipse.cpu().numpy().squeeze().astype("float64"),
            X.cpu().numpy().squeeze().astype("float64"),
        )

        # Visualize the result
        visualization_with_zoom(
            example,
            ell1_ellipse.cpu().numpy().squeeze().astype("float64"),
            zoom,
            colorbar,
            "results/figures/ellipse/0000-ellipse_ell1.pdf",
        )
        print("##############################################################")
        # model_names = []
    print(f"Models: {model_names}")
    for model_name in model_names:

        if model_name.split("_")[0] == "tv":
            Y = tv_ellipse
        elif model_name.split("_")[0] == "ell1":
            Y = ell1_ellipse
        elif model_name.split("_")[0] == "landweber":
            Y = landweber_ellipse
        else:
            Y = fbp_ellipse
        Z = data_limited

        X = torch.Tensor(X).to(config.device)
        Y = torch.Tensor(Y).to(config.device)
        Z = torch.Tensor(Z).to(config.device)

        X = X.unsqueeze(0).unsqueeze(0)
        Y = Y.squeeze().unsqueeze(0).unsqueeze(0)
        Z = Z.unsqueeze(0).unsqueeze(0)

        print(f"Model: {model_name}.")

        radon_full, radon_limited, radon_null_space = get_radon_operators(example)

        model = UNet(1, 1, model_name, example)
        model.load_state_dict(torch.load(f"models/{example}/{model_name}.pth"))
        model.to(config.device)
        model_output = model(Y)

        # model_output, _ = calc_output(
        #     model_name, model, X, Y, Z,
        #     radon_full, radon_limited, None, config
        # )

        diff_recons = abs(
            radon_limited.forward(model_output) - radon_limited.forward(Y)
        )  # model_output - Y
        radon_diff_recons = diff_recons  # radon_full.forward(diff_recons)
        diff_plots[f"results/figures/ellipse/0000-ellipse_diff_{model_name}.pdf"] = abs(
            radon_diff_recons.cpu().detach().numpy().squeeze().astype("float64")
        )
        errors_data[f"{model_name}_data"] = calc_errors(
            radon_limited.forward(model_output)
            .cpu()
            .detach()
            .numpy()
            .squeeze()
            .astype("float64"),
            radon_limited.forward(Y).cpu().numpy().squeeze().astype("float64"),
        )
        mse_error_diff = errors_data[f"{model_name}_data"]["MSE"]
        figure_names_diff.append(f"0000-ellipse_diff_{model_name}.pdf")
        captions_diff.append(
            f"|\\forward(\\signal_{{\\rm {model_name.split('_')[0].upper()}}}^{{{model_name.split('_')[-1].upper()}}}) - \\forward(\\signal_{{\\rm {model_name.split('_')[0].upper()}}})|, \\text{{MSE}}: {round(mse_error_diff, 4)}"
        )
        labels_diff.append(f"0000-ellipse_diff_{model_name}.pdf")

        model_output = model_output.cpu().detach().numpy().squeeze().astype("float64")

        X = X.cpu().numpy().squeeze()

        print_errors(model_output, X)

        print("Plotting.")
        visualization_with_zoom(
            example,
            model_output,
            zoom,
            colorbar,
            f"results/figures/ellipse/0000-ellipse_{model_name}.pdf",
        )
        print(f"Finished {model_name}.")

        errors[model_name] = calc_errors(model_output, X)
        print("##############################################################")

    table_path = "results/ellipse_error_table.csv"
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

    print(generate_latex_figure_block(figure_names_diff, captions_diff, labels_diff))
    latex_figure_diff = generate_latex_figure_block(
        figure_names_diff, captions_diff, labels_diff
    )
    with open("results/ellipse_latex_diff_figure.txt", "w") as f:
        f.write(latex_figure_diff)

    create_error_table(
        errors, f"results/ellipse_recon_error_table.csv", "ellipse", False
    )

    print("Finished.")


if __name__ == "__main__":
    import argparse

    from src.utils.load_config import load_config
    from src.utils.parser import parse_nested_list

    parser = argparse.ArgumentParser(description="Testing inputs.")
    parser.add_argument("example", type=str, help="Which example.")
    parser.add_argument("model_names", type=parse_nested_list, nargs="?")

    # Parse arguments from the command line
    args = parser.parse_args()

    if args.model_names is None:
        from example import models

        model_names = models

    config = load_config(args.example)
    print(f"Testing for example {args.example}")
    print(f"Models: {model_names}")
    print("##############################################################")
    test_other_function_class(example=args.example, model_names=model_names)
    print(f"Testing completed for {args.example}.")
