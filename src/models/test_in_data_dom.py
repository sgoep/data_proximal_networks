import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage.metrics import mean_squared_error as mse

from src.data.data_loader import DataLoader
from src.models.unet import UNet
from src.models.utils import extension_with_zero
from src.utils.load_config import load_config
from src.utils.parser import parse_nested_list
from src.utils.radon_operator import get_radon_operators
from src.utils.test_utils import method_latex_mapping
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


def test(example, model_names):
    config = load_config(example)

    chi = np.zeros([config.num_angles_full, config.det_count])
    chi[config.angles_full <= config.phi_limited] = 1

    radon_full, radon_limited, radon_null_space = get_radon_operators(example)
    index = 1366
    plotted = False

    # Process reconstructions
    methods = [("landweber", False), ("tv", True), ("ell1", True)]
    errors_data1 = {}
    errors_data2 = {}
    errors_data3 = {}

    for initial_regul, initrecon in methods:
        print(f"Model: {initial_regul}.")
        X, Y, Z = get_sample(
            index, example, initrecon, initial_regul, "validation", config
        )

        if not plotted:
            radon_gt = radon_full.forward(X).cpu().numpy().squeeze().astype("float64")
            np.save(f"results/figures/0000-{example}_ground_truth_data", radon_gt)

            plt.figure()
            plt.imshow(radon_gt, cmap="gray")
            plt.axis("off")
            plt.savefig(
                f"results/figures/0000-{example}-data_ground_truth.pdf",
                bbox_inches="tight",
            )

            visualization_with_zoom(
                example,
                X.cpu().numpy().squeeze().astype("float64"),
                False,
                False,
                f"results/figures/{example}/0000-{example}_ground_truth.pdf",
            )
            plotted = True
            np.save(
                f"results/figures/0000-{example}_ground_truth_image",
                X.cpu().numpy().squeeze().astype("float64"),
            )

        radon_init_recon = (
            radon_full.forward(Y).cpu().numpy().squeeze().astype("float64")
        )
        plt.figure()
        plt.imshow(radon_init_recon, cmap="gray")
        plt.axis("off")
        plt.savefig(
            f"results/figures/0000-{example}-data_{initial_regul}.pdf",
            bbox_inches="tight",
        )

        plt.figure()
        plt.imshow(abs(radon_init_recon - radon_gt), cmap="gray")
        plt.axis("off")
        plt.savefig(
            f"results/figures/0000-{example}-data_{initial_regul}_diff.pdf",
            bbox_inches="tight",
        )

        errors_data1[method_latex_mapping[initial_regul]] = mse(
            radon_full.forward(Y).cpu().numpy().squeeze().astype("float64") * chi,
            extension_with_zero(Z, config).cpu().numpy().squeeze().astype("float64"),
        )
        errors_data2[method_latex_mapping[initial_regul]] = mse(
            radon_full.forward(Y).cpu().numpy().squeeze().astype("float64") * chi,
            radon_full.forward(X).cpu().numpy().squeeze().astype("float64") * chi,
        )
        errors_data3[method_latex_mapping[initial_regul]] = mse(
            radon_full.forward(Y).cpu().numpy().squeeze().astype("float64") * (1 - chi),
            radon_full.forward(X).cpu().numpy().squeeze().astype("float64") * (1 - chi),
        )

        np.save(
            f"results/figures/0000-{example}_{initial_regul}_image",
            Y.cpu().numpy().squeeze().astype("float64"),
        )
        np.save(
            f"results/figures/0000-{example}_{initial_regul}_data", radon_init_recon
        )

    for model_name in model_names:
        print(f"Model: {model_name}.")
        initial_regul = model_name.split("_")[0]
        initrecon = initial_regul in ["tv", "ell1"]

        X, Y, Z = get_sample(
            index, example, initrecon, initial_regul, "validation", config
        )

        model = UNet(1, 1, model_name, example).to(config.device)
        model.load_state_dict(torch.load(f"models/{example}/{model_name}.pth"))
        model_output = model(Y).detach()

        radon_recon = (
            radon_full.forward(model_output)
            .cpu()
            .detach()
            .numpy()
            .squeeze()
            .astype("float64")
        )

        plt.figure()
        plt.imshow(radon_recon, cmap="gray")
        plt.axis("off")
        plt.savefig(
            f"results/figures/0000-{example}-data_{model_name}.pdf", bbox_inches="tight"
        )

        plt.figure()
        plt.imshow(abs(radon_recon - radon_gt), cmap="gray")
        plt.axis("off")
        plt.savefig(
            f"results/figures/0000-{example}-data_{model_name}_diff.pdf",
            bbox_inches="tight",
        )

        errors_data1[method_latex_mapping[model_name]] = mse(
            radon_full.forward(model_output).cpu().numpy().squeeze().astype("float64")
            * chi,
            extension_with_zero(Z, config).cpu().numpy().squeeze().astype("float64"),
        )
        errors_data2[method_latex_mapping[model_name]] = mse(
            radon_full.forward(model_output).cpu().numpy().squeeze().astype("float64")
            * chi,
            radon_full.forward(X).cpu().numpy().squeeze().astype("float64") * chi,
        )
        errors_data3[method_latex_mapping[model_name]] = mse(
            radon_full.forward(model_output).cpu().numpy().squeeze().astype("float64")
            * (1 - chi),
            radon_full.forward(X).cpu().numpy().squeeze().astype("float64") * (1 - chi),
        )

        np.save(
            f"results/figures/0000-{example}_{model_name}_image",
            model_output.cpu().numpy().squeeze().astype("float64"),
        )
        np.save(f"results/figures/0000-{example}_{model_name}_data", radon_recon)

    print(errors_data1)
    print(errors_data2)
    print(errors_data3)

    errors = {}
    errors["$\\radon_\\Omega (\\signal_{{\\rm REC}}) - \\data^\\delta$"] = (
        errors_data1  # [val for _, val in errors_data1.items()]
    )
    errors[
        "$\\radon_\\Omega (\\signal_{{\\rm REC}}) - \\radon_\\Omega (\\signal^\\star)$"
    ] = errors_data2  # [val for _, val in errors_data3.items()]
    errors[
        "$\\radon_{{\\Omega^c}} (\\signal_{{\\rm REC}}) - \\radon_{{\\Omega^c}} (\\signal^\\star)$"
    ] = errors_data3  # [val for _, val in errors_data3.items()]

    df = pd.DataFrame(
        {
            "$\\radon_\\Omega (\\signal_{{\\rm REC}}) - \\data^\\delta$": errors_data1,
            "$\\radon_\\Omega (\\signal_{{\\rm REC}}) - \\radon_\\Omega (\\signal^\\star)$": errors_data2,
            "$\\radon_{{\\Omega^c}} (\\signal_{{\\rm REC}}) - \\radon_{{\\Omega^c}} (\\signal^\\star)$": errors_data3,
        }
    )
    df.to_csv("results/figures/0000-errors-data.csv", index_label="Index")

    latex_output = df.to_latex(
        index=True,
        formatters={"name": str.upper},
        float_format="{:.4f}".format,
    )
    print(latex_output)
    # print(pd.DataFrame.from_dict(errors_data1, index=[0]))
    # print(pd.DataFrame.from_dict(errors_data2, index=[0]))
    # print(pd.DataFrame.from_dict(errors_data3, index=[0]))


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
