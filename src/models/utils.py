import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from example import example
from src.data.data_loader import DataLoader
from src.models.unet import UNet


def data_prox_func(X: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    Applies a proximity operator to the input tensor `X` based on a threshold
    `beta`.

    Args:
        X (torch.Tensor): The input tensor, expected to be an element in the
        range of the limited angle Radon transform.
        beta (torch.Tensor): A scalar tensor used as the threshold for the
        proximity operation. It determines the maximum allowable norm for the
        slices.

    Returns:
        torch.Tensor: A tensor `Y` with the same shape as `X`, containing the
        processed slices.

    """
    Y = torch.zeros_like(X).to("cuda")
    for j in range(Y.shape[0]):
        norm = torch.linalg.norm(X[j, 0, :, :])
        if norm <= beta:
            Y[j, 0, :, :] = X[j, 0, :, :]
        else:
            Y[j, 0, :, :] = beta * X[j, 0, :, :] / norm

    return Y

    # avg_max = torch.Tensor(
    #         np.load("data/data_synthetic/avg_max.npy")
    #     ).to(config.device)
    # Y = X.clone()
    # Y[torch.abs(Y) >= avg_max] = avg_max
    # return Y


def extension_with_zero(X: torch.Tensor, config) -> torch.Tensor:
    return F.pad(X, (0, 0, 0, config.num_angles_full - config.num_angles_limited))


def init_weights(m):

    torch.manual_seed(0)
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0.0, std=np.sqrt(2 / (3**2 * m.in_channels)))
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=np.sqrt(2 / m.in_features))
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def save_weights_as_json(model, file_name):
    weights_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            weights_dict[name] = param.data.tolist()

    with open(file_name, "w") as f:
        json.dump(weights_dict, f)


def load_weights_from_json(model, file_name):
    with open(file_name, "r") as f:
        weights_dict = json.load(f)

    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = torch.tensor(weights_dict[name])


def plot_losses(train_loss, test_loss, name):
    """
    This function plots the training and validation losses and saves the
    figure as a PDF.

    Args:
        train_loss (list): List of training losses for the model.
        test_loss (list): List of validation losses for the model.
        name (str): The name of the model, used in the plot label and filename.
    """

    plt.figure()

    # Plot training and validation losses
    plt.plot(train_loss, marker="x", label="Training Loss")
    plt.plot(test_loss, marker="x", label="Validation Loss")

    # Set the y-axis label and other plot properties
    plt.ylabel(f"{name}", fontsize=22)
    plt.legend()

    # Save the figure as a PDF
    plt.savefig(f"{name}")
    plt.close()  # Close the figure to free up memory


def get_data_loader(example, initrecon, initial_regul, config):
    model_train_set = DataLoader(
        id=[i for i in range(config.training_params["len_train"])],
        initrecon=initrecon,
        test_train="train",
        example=example,
        initial_regul=initial_regul,
    )

    model_test_set = DataLoader(
        id=[i for i in range(config.training_params["len_test"])],
        initrecon=initrecon,
        test_train="test",
        example=example,
        initial_regul=initial_regul,
    )

    TrainDataGen = torch.utils.data.DataLoader(model_train_set, **config.model_params)
    TestDataGen = torch.utils.data.DataLoader(model_test_set, **config.model_params)
    return TrainDataGen, TestDataGen


def get_model(training_params, model_name, example):
    model = UNet(n_channels=1, n_classes=1, model_name=model_name, example=example)
    model.apply(init_weights)
    model.to("cuda")

    model_optimizer = optim.Adam(
        model.parameters(), lr=training_params["learning_rate"]
    )
    model_error = nn.MSELoss()
    return model, model_optimizer, model_error
