import numpy as np
import torch

from src.models.utils import get_data_loader, get_model, plot_losses
from src.utils.radon_operator import get_radon_operators


def train(example: str, model_name: str, config):
    if model_name.split("_")[0] == "fbp":
        initrecon = False
        initial_regul = "fbp"
    elif model_name.split("_")[0] == "landweber":
        initrecon = False
        initial_regul = "landweber"
    elif model_name.split("_")[0] in ["tv", "ell1"]:
        initrecon = True
        initial_regul = model_name.split("_")[0]
    else:
        raise ValueError(f"Unknown model {model_name}.")

    if config.factor is not None:
        print(f"Start training of {model_name} with factor {config.factor}.")
    else:
        print(f"Start training of {model_name}, initial recon: {initial_regul}.")

    best_test_loss = float("inf")
    best_model_params = None

    model, optimizer, error = get_model(config.training_params, model_name, example)
    TrainDataGen, TestDataGen = get_data_loader(
        example, initrecon, initial_regul, config
    )

    best_test_loss = float("inf")
    best_model_params = None

    torch.cuda.empty_cache()

    train_loss_list = np.zeros(config.epochs)
    test_loss_list = np.zeros(config.epochs)

    for epoch in np.arange(1, config.epochs + 1):
        train_loss = 0.0
        test_loss = 0.0

        model.train()
        for X, Y, Z in TrainDataGen:

            X = X.to(device=config.device, dtype=torch.float)
            Y = Y.to(device=config.device, dtype=torch.float)
            output = model(Y)
            loss = error(output.double(), X.double())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.data.item()

        with torch.no_grad():
            model.eval()
            for X, Y, Z in TestDataGen:

                X = X.to(device=config.device, dtype=torch.float)
                Y = Y.to(device=config.device, dtype=torch.float)
                output = model(Y)

                loss = error(output.double(), X.double())

                test_loss += loss.data.item()

            train_loss = train_loss / config.len_train
            test_loss = test_loss / config.len_test

            if test_loss < best_test_loss:
                best_epoch = epoch
                best_test_loss = test_loss
                best_model_params = model.state_dict().copy()
                if config.factor is not None:
                    torch.save(
                        best_model_params,
                        f"models/{example}/{model_name}_factor_{config.factor}.pth",
                    )
                else:
                    torch.save(best_model_params, f"models/{example}/{model_name}.pth")

        print(
            f"Epoch [{epoch}/{config.epochs}], Train Loss: {train_loss:.8f} "
            f"and Test Loss: {test_loss:.8f}"
        )

        train_loss_list[epoch - 1] = train_loss
        test_loss_list[epoch - 1] = test_loss
        if config.factor is not None:
            plot_losses(
                train_loss_list[1:],
                test_loss_list[1:],
                f"losses/{example}_loss_{model_name}_factor_{config.factor}.pdf",
            )
        else:
            plot_losses(
                train_loss_list[1:],
                test_loss_list[1:],
                f"losses/{example}_loss_{model_name}.pdf",
            )

    print(f"Saving model {model_name}, best epoch: {best_epoch}.")
    if config.factor is not None:
        torch.save(
            best_model_params,
            f"models/{example}/{model_name}_factor_{config.factor}.pth",
        )
    else:
        torch.save(best_model_params, f"models/{example}/{model_name}.pth")

    print("############################################################## ")


if __name__ == "__main__":
    import argparse

    from src.utils.load_config import load_config
    from src.utils.parser import parse_nested_list

    parser = argparse.ArgumentParser(description="Training inputs.")
    parser.add_argument("example", type=str, help="Which example.")
    parser.add_argument("model_names", type=parse_nested_list, nargs="?")

    # Parse arguments from the command line
    args = parser.parse_args()

    if args.model_names is None:
        from example import models

        model_names = models

    config = load_config(args.example)

    print(f"Training for example {args.example}")
    print(f"Models: {model_names}")
    print("##############################################################")
    print("Configs:")
    print(f"Size training set: {config.training_params['len_train']}")
    print(f"Size test set: {config.training_params['len_test']}")
    print(f"Epochs: {config.training_params['epochs']}")
    print(f"Learning rate: {config.training_params['learning_rate']}")
    print(f"Training on: {config.device}")
    print("##############################################################")

    for model_name in model_names:
        train(
            example=args.example,
            model_name=model_name,
            # initial_regul=model_name[1],
            config=config,
        )
    print(f"Training completed for {args.example}.")
