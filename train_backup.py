from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore

from example import example, models

# from config import config
from src.data.data_loader import DataLoader
from src.models.unet import UNet
from src.models.utils import (
    compute_learned_output,
    init_weights,
    load_weights_from_json,
    save_weights_as_json,
)
from src.utils.load_config import load_config
from src.utils.radon_operator import get_radon_operators

config = load_config()


def train(
    model,
    TrainDataGenerator,
    TestDataGenerator,
    error,
    optimizer,
    device,
    NumEpochs,
    len_train,
    len_test,
    model_name,
    config,
) -> Tuple[Dict, List, List]:

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    best_test_loss = float("inf")
    best_model_params = None

    torch.cuda.empty_cache()
    # gc.collect()

    history = {}
    history["loss"] = []
    history["test_loss"] = []
    train_loss_list = np.zeros(NumEpochs)
    test_loss_list = np.zeros(NumEpochs)

    radon_full, radon_limited = get_radon_operators(example)

    if model_name in ["fbp_dp", "tv_dp"]:
        beta = config.norm
        # torch.Tensor(
        # np.load(f"data/data_{example}/data_processed/train/norm.npy")
        # ).to(config.device)

    for epoch in np.arange(1, NumEpochs + 1):
        train_loss = 0.0
        test_loss = 0.0

        model.train()
        for X, Y, Z in TrainDataGenerator:
            optimizer.zero_grad()

            X = X.to(device=config.device, dtype=torch.float)
            Y = Y.to(device=config.device, dtype=torch.float)
            Z = Z.to(device=config.device, dtype=torch.float)

            output = model(Y)

            learned_output, X_output = compute_learned_output(
                model_name=model_name,
                output=output,
                radon_full=radon_full,
                X=X,
                Y=Y,
                Z=Z,
                beta=beta if model_name in ["fbp_dp", "tv_dp"] else None,
            )

            loss = error(learned_output.double(), X_output.double())

            loss.backward()
            optimizer.step()

            train_loss += loss.data.item()
            history["loss"].append(train_loss)

        model.eval()
        for X, Y, Z in TestDataGenerator:

            X = X.to(device=device, dtype=torch.float)
            Y = Y.to(device=device, dtype=torch.float)
            Z = Z.to(device=device, dtype=torch.float)

            output = model(Y)

            learned_output, X_output = compute_learned_output(
                model_name=model_name,
                output=output,
                radon_full=radon_full,
                X=X,
                Y=Y,
                Z=Z,
                beta=beta if model_name in ["fbp_dp", "tv_dp"] else None,
            )

            loss = error(learned_output.double(), X_output.double())

            test_loss += loss.data.item()
            history["test_loss"].append(test_loss)

        train_loss = train_loss / len_train
        test_loss = test_loss / len_test

        # Step the scheduler
        # scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        if test_loss < best_test_loss:
            best_epoch = epoch
            best_test_loss = test_loss
            best_model_params = model.state_dict().copy()
            torch.save(best_model_params, f"models/{example}/{model_name}.pth")

        print(
            f"Epoch {epoch}/{NumEpochs}, Train Loss: {train_loss:.8f} and "
            f"Test Loss: {test_loss:.8f}, learning rate: {lr}"
        )

        train_loss_list[epoch - 1] = train_loss
        test_loss_list[epoch - 1] = test_loss

    print("######################## Saving model ######################## ")
    print(f"###### {model_name}, epoch: {best_epoch} ######")
    torch.save(best_model_params, f"models/{example}/{model_name}.pth")

    return history, train_loss_list, test_loss_list


def start_training(
    initrecon: bool,
    model_name: str,
    model_params: dict,
    training_params: dict,
    config,
) -> None:

    # torch.cuda.manual_seed(42)
    # torch.cuda.manual_seed_all(42)

    def get_model(training_params):
        model = UNet(
            n_channels=1,
            n_classes=1,
        )
        model.apply(init_weights)
        model.to(config.device)

        model_optimizer = optim.Adam(
            model.parameters(), lr=training_params["learning_rate"]
        )
        model_error = nn.MSELoss()
        return model, model_optimizer, model_error

    model, optimizer, error = get_model(training_params)

    # model_ran.apply(init_weights)

    # if Path("models/initial_weights.json").is_file():
    #     load_weights_from_json(model, "models/initial_weights.json")
    # else:
    #     model.apply(init_weights)
    #     save_weights_as_json(model, "models/initial_weights.json")

    # model.to(config.device)

    # model_optimizer = optim.Adam(
    #     model.parameters(), lr=training_params["learning_rate"]
    # )
    # model_error = nn.MSELoss()

    model_train_set = DataLoader(
        id=[i for i in range(training_params["len_train"])],
        initrecon=initrecon,
        test_train="train",
        example=example,
        # which=config.initial_regularization_method
    )

    model_test_set = DataLoader(
        id=[i for i in range(training_params["len_test"])],
        initrecon=initrecon,
        test_train="test",
        example=example,
        # which=config.initial_regularization_method
    )

    TrainDataGen = torch.utils.data.DataLoader(model_train_set, **model_params)
    TestDataGen = torch.utils.data.DataLoader(model_test_set, **model_params)

    print("##################### Start training ######################### ")
    print(f"Example: {example}")
    print(f"Model: {model_name}")
    print(f"Size training set: {training_params['len_train']}")
    print(f"Size test set: {training_params['len_test']}")
    print(f"Epochs: {training_params['epochs']}")
    print(f"Learning rate: {training_params['learning_rate']}")
    print(f"Training on: {config.device}")
    print("############################################################## ")

    _, train_loss_list, test_loss_list = train(
        model,
        TrainDataGen,
        TestDataGen,
        error,
        optimizer,
        config.device,
        training_params["epochs"],
        training_params["len_train"],
        training_params["len_test"],
        model_name,
        config,
    )

    loss_plot = "losses/loss_" + model_name + ".pdf"

    plt.figure()
    plt.plot(train_loss_list[1:], marker="x", label="Training Loss")
    plt.plot(test_loss_list[1:], marker="x", label="Validation Loss")
    plt.ylabel("loss_" + model_name, fontsize=22)
    plt.legend()
    plt.savefig(loss_plot)

    print("########################## Finished ########################## ")


def training(example: str):
    print("Start training.")

    model_params = {"batch_size": 16, "shuffle": True, "num_workers": 2}
    training_params = {
        "learning_rate": config.learning_rate,
        "len_train": config.len_train,
        "len_test": config.len_test,
        "epochs": config.epochs,
    }

    # List of model configurations
    model_configs = [
        # {"initrecon": False, "model_name": "fbp_X_res"},
        # {"initrecon": False, "model_name": "fbp_res"},
        {"initrecon": False, "model_name": "fbp_nsn"},
        # {"initrecon": False, "model_name": "fbp_dp"},
        # {"initrecon": True, "model_name": "tv_X_res"},
        # {"initrecon": True, "model_name": "tv_res"},
        # {"initrecon": True, "model_name": "tv_nsn"},
        # {"initrecon": True, "model_name": "tv_dp"}
    ]

    # Loop through the configurations and call start_training
    for config_dict in model_configs:
        start_training(
            initrecon=config_dict["initrecon"],
            model_name=config_dict["model_name"],
            model_params=model_params,
            training_params=training_params,
            config=config,
        )

    print("End of training. ")


if __name__ == "__main__":
    training(example=example)
