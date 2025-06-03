import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize

from src.data.data_loader import DataLoader
from src.models.unet import UNet
from src.utils.load_config import load_config
from src.utils.parser import parse_nested_list
from src.utils.radon_operator import get_radon_operators
from src.utils.test_utils import calc_errors, create_error_table, print_errors
from src.visualization.latex_figure import generate_latex_figure_block
from src.visualization.visualization import visualization_with_zoom

# Config options
zoom, colorbar = True, False


def normalize_data(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))


def get_sample(index, example, initrecon, initial_regul, test_train, config):
    D = DataLoader(
        np.arange(0, index + 1),
        initrecon=initrecon,
        example=example,
        initial_regul=initial_regul,
        test_train=test_train,
    )
    X, Y, Z = D[index]
    return (
        torch.Tensor(X).to(config.device).unsqueeze(0),
        torch.Tensor(Y).to(config.device).unsqueeze(0),
        torch.Tensor(Z).to(config.device),
    )


def process_reconstruction(
    index, example, initrecon, initial_regul, test_train, label, config, radon_limited
):
    D = DataLoader(
        np.arange(0, index + 1),
        initrecon=initrecon,
        example=example,
        initial_regul=initial_regul,
        test_train=test_train,
    )
    X, Y, _ = D[index]
    print_errors(Y.squeeze(), X.squeeze())
    visualization_with_zoom(
        example,
        Y.squeeze(),
        zoom,
        colorbar,
        f"results/figures/{example}/0000-{example}_{label}.pdf",
    )

    return calc_errors(Y, X), calc_errors(
        radon_limited.forward(Y).cpu().numpy().squeeze(),
        radon_limited.forward(X).cpu().numpy().squeeze(),
    )


def test(example, model_names):
    config = load_config(example)
    radon_full, radon_limited, radon_null_space = get_radon_operators(example)
    index = 1366
    # index = 10
    plotted = False

    figure_names = []
    captions = []
    labels = []

    figure_names_diff = []
    captions_diff = []
    labels_diff = []

    # Process reconstructions
    methods = [
        # ("fbp", False),
        ("landweber", False),
        ("tv", True),
        ("ell1", True),
    ]
    errors, errors_data = {}, {}
    diff_plots = {}
    for initial_regul, initrecon in methods:
        print(f"Model: {initial_regul}.")
        X, Y, Z = get_sample(
            index, example, initrecon, initial_regul, "validation", config
        )
        errors[initial_regul] = calc_errors(
            Y.cpu().numpy().squeeze().astype("float64"),
            X.cpu().numpy().squeeze().astype("float64"),
        )
        errors_data[f"{initial_regul}_data"] = calc_errors(
            radon_limited.forward(Y).cpu().numpy().squeeze().astype("float64"),
            radon_limited.forward(X).cpu().numpy().squeeze().astype("float64"),
        )

        print_errors(
            Y.cpu().numpy().squeeze().astype("float64"),
            X.cpu().numpy().squeeze().astype("float64"),
        )
        visualization_with_zoom(
            example,
            Y.cpu().numpy().squeeze().astype("float64"),
            zoom,
            colorbar,
            f"results/figures/{example}/0000-{example}_{initial_regul}.pdf",
        )

        figure_names.append(f"0000-{example}_{initial_regul}.pdf")
        captions.append(f"$\\signal_{{\\rm {initial_regul.upper()}}}$")
        labels.append(f"0000-{example}_{initial_regul}.pdf")

        diff_recons = radon_full.forward(Y - X)
        radon_diff_recons = diff_recons
        # radon_diff_recons = abs(radon_full.forward(diff_recons))

        diff_plots[
            f"results/figures/{example}/0000-{example}_diff_{initial_regul}.pdf"
        ] = abs(radon_diff_recons.cpu().detach().numpy().squeeze().astype("float64"))
        # figure_names_diff.append(f"0000-{example}_diff_{initial_regul}.pdf")
        # captions_diff.append(
        #     f"\\forward(\\signal_{{\\rm {initial_regul.upper()}}} - \\signal^\\star)"
        # )
        # labels.append(f"0000-{example}_diff_{initial_regul}.pdf")

        if not plotted:
            visualization_with_zoom(
                example,
                X.cpu().numpy().squeeze().astype("float64"),
                False,
                False,
                f"results/figures/{example}/0000-{example}_ground_truth.pdf",
            )
            plotted = True

        print(f"Finished {initial_regul}.")
        print("##############################################################")

    for model_name in model_names:
        print(f"Model: {model_name}.")
        initial_regul = model_name.split("_")[0]
        initrecon = initial_regul in ["tv", "ell1"]

        X, Y, Z = get_sample(
            index, example, initrecon, initial_regul, "validation", config
        )

        model = UNet(1, 1, model_name, example).to(config.device)
        model.load_state_dict(torch.load(f"models/{example}/{model_name}.pth"))

        model_output = model(Y)

        diff_recons = radon_full.forward(model_output - X)
        radon_diff_recons = diff_recons
        # radon_diff_recons = abs(radon_full.forward(diff_recons))

        diff_plots[
            f"results/figures/{example}/0000-{example}_diff_{model_name}.pdf"
        ] = abs(radon_diff_recons.cpu().detach().numpy().squeeze().astype("float64"))
        errors[model_name] = calc_errors(
            model_output.cpu().detach().numpy().squeeze(), X.cpu().numpy().squeeze()
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
        figure_names_diff.append(f"0000-{example}_diff_{model_name}.pdf")
        captions_diff.append(
            f"|\\forward(\\signal_{{\\rm {initial_regul.upper()}}}^{{{model_name.split('_')[-1].upper()}}}) - \\forward(\\signal_{{\\rm {initial_regul.upper()}}})|, \\text{{MSE}}: {round(mse_error_diff, 4)}"
        )
        labels_diff.append(f"0000-{example}_diff_{model_name}.pdf")

        print_errors(
            model_output.cpu().detach().numpy().squeeze(),
            X.cpu().detach().numpy().squeeze(),
        )
        visualization_with_zoom(
            example,
            model_output.cpu().detach().numpy().squeeze(),
            zoom,
            colorbar,
            f"results/figures/{example}/0000-{example}_{model_name}.pdf",
        )

        figure_names.append(f"0000-{example}_{model_name}.pdf")
        captions.append(
            f"$\\signal_{{\\rm {initial_regul}}}^{{\\rm {model_name.split('_')[-1].upper()}}}$"
        )
        labels.append(f"0000-{example}_{model_name}.pdf")

        print(f"Finished {model_name}.")
        print("##############################################################")

    # Normalize and save error plots
    norm = Normalize(
        vmin=min(v.min() for v in diff_plots.values()),
        vmax=max(v.max() for v in diff_plots.values()),
    )
    for path, val in diff_plots.items():
        plt.figure()
        plt.imshow(val, norm=norm, cmap="gray")
        plt.axis("off")
        # plt.colorbar()x
        plt.savefig(path, bbox_inches="tight")

    create_error_table(
        errors, f"results/{example}_recon_error_table.csv", example, False
    )
    create_error_table(
        errors_data, f"results/{example}_data_error_table.csv", example, True
    )

    print(generate_latex_figure_block(figure_names, captions, labels))
    latex_figure = generate_latex_figure_block(figure_names, captions, labels)
    with open(f"results/{example}_latex_figure.txt", "w") as f:
        f.write(latex_figure)

    print(generate_latex_figure_block(figure_names_diff, captions_diff, labels_diff))
    latex_figure_diff = generate_latex_figure_block(
        figure_names_diff, captions_diff, labels_diff
    )
    with open(f"results/{example}_latex_diff_figure.txt", "w") as f:
        f.write(latex_figure_diff)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing inputs.")
    parser.add_argument("example", type=str, help="Which example.")
    parser.add_argument("model_names", type=parse_nested_list, nargs="?")

    args = parser.parse_args()
    if args.model_names is None:
        from example import models

        model_names = models
    print(f"Testing for example {args.example}")
    print(f"Models: {model_names}")
    print("##############################################################")
    test(example=args.example, model_names=model_names)
    print(f"Testing completed for {args.example}.")
