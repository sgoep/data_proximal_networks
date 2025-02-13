from typing import List

import numpy as np  # type: ignore
import torch  # type: ignore

# from config import config
from src.models.model_output import (calc_dp_output, calc_dp_X_output,
                                     calc_nsn_output, calc_res_output,
                                     calc_single_dp_output)
from src.models.utils import get_data_loader, get_model, plot_losses
from src.utils.radon_operator import filter_sinogram, get_radon_operators

# config = load_config(example)


def train_dp(model_name: str, initial_regul: str, config, example):
    print(f"Start training of {model_name}.")

    if model_name in ["fbp_dp", "fbp_X_dp"]:
        initrecon = False
    elif model_name in ["tv_dp", "ell1_dp", "tv_X_dp", "ell1_X_dp"]:
        initrecon = True
    else:
        raise ValueError(f"Unknown model {model_name}.")

    model_ran, optimizer_ran, error_ran = get_model(
        config.training_params
    )
    model_ker, optimizer_ker, error_ker = get_model(
        config.training_params
    )

    TrainDataGen, TestDataGen = get_data_loader(initrecon, initial_regul)

    radon_full, radon_limited, _ = get_radon_operators(example)

    best_test_loss_ran = float('inf')
    best_test_loss_ker = float('inf')
    best_model_params_ran = None
    best_model_params_ker = None

    torch.cuda.empty_cache()

    train_loss_list_ran = np.zeros(config.epochs)
    test_loss_list_ran = np.zeros(config.epochs)

    train_loss_list_ker = np.zeros(config.epochs)
    test_loss_list_ker = np.zeros(config.epochs)

    for epoch in np.arange(1, config.epochs+1):
        train_loss_ran = 0.0
        train_loss_ker = 0.0
        test_loss_ran = 0.0
        test_loss_ker = 0.0

        model_ran.train()
        model_ker.train()

        for X, Y, Z in TrainDataGen:

            X = X.to(device=config.device, dtype=torch.float)
            Y = Y.to(device=config.device, dtype=torch.float)
            Z = Z.to(device=config.device, dtype=torch.float)

            if model_name in ["fbp_dp", "tv_dp", "ell1_dp"]:
                (model_output_ran, X_output_ran,
                    model_output_ker, X_output_ker) = calc_dp_output(
                    model_name, model_ran, model_ker,
                    X, Y, Z, radon_full, radon_limited, config
                )
            elif model_name in ["fbp_X_dp", "tv_X_dp", "ell1_X_dp"]:
                (model_output_ran, X_output_ran,
                    model_output_ker, X_output_ker) = calc_dp_X_output(
                    model_name, model_ran, model_ker,
                    X, Y, Z, radon_full, radon_limited, config
                )
            else:
                raise ValueError(f"Unknown model {model_name}.")

            loss_ran = error_ran(
                model_output_ran.double(), X_output_ran.double()
            )
            loss_ker = error_ker(
                model_output_ker.double(), X_output_ker.double()
            )

            loss_ran.backward()
            optimizer_ran.step()
            optimizer_ran.zero_grad()

            loss_ker.backward()
            optimizer_ker.step()
            optimizer_ker.zero_grad()

            train_loss_ran += loss_ran.data.item()
            train_loss_ker += loss_ker.data.item()
            # history["loss"].append(train_loss_ran)

        with torch.no_grad():
            model_ran.eval()
            model_ker.eval()
            for X, Y, Z in TestDataGen:

                X = X.to(device=config.device, dtype=torch.float)
                Y = Y.to(device=config.device, dtype=torch.float)
                Z = Z.to(device=config.device, dtype=torch.float)

                if model_name in ["fbp_dp", "tv_dp", "ell1_dp"]:
                    (model_output_ran, X_output_ran,
                        model_output_ker, X_output_ker) = calc_dp_output(
                        model_name, model_ran, model_ker,
                        X, Y, Z, radon_full, radon_limited, config
                    )
                elif model_name in ["fbp_X_dp", "tv_X_dp", "ell1_X_dp"]:
                    (model_output_ran, X_output_ran,
                        model_output_ker, X_output_ker) = calc_dp_X_output(
                        model_name, model_ran, model_ker,
                        X, Y, Z, radon_full, radon_limited, config
                    )
                else:
                    raise ValueError(f"Unknown model {model_name}.")

                loss_ran = error_ran(
                    model_output_ran.double(), X_output_ran.double()
                )
                loss_ker = error_ker(
                    model_output_ker.double(), X_output_ker.double()
                )

                test_loss_ran += loss_ran.data.item()
                test_loss_ker += loss_ker.data.item()

            train_loss_ran = train_loss_ran/config.len_train
            test_loss_ran = test_loss_ran/config.len_test

            train_loss_ker = train_loss_ker/config.len_train
            test_loss_ker = test_loss_ker/config.len_test

            if test_loss_ran < best_test_loss_ran:
                best_epoch_ran = epoch
                best_test_loss_ran = test_loss_ran
                best_model_params_ran = model_ran.state_dict().copy()
                torch.save(
                    best_model_params_ran,
                    f"models/{example}/{model_name}_ran.pth"
                )

            if test_loss_ker < best_test_loss_ker:
                best_epoch_ker = epoch
                best_test_loss_ker = test_loss_ker
                best_model_params_ker = model_ker.state_dict().copy()
                torch.save(
                    best_model_params_ker,
                    f"models/{example}/{model_name}_ker.pth"
                )

        print(
            f"Epoch [{epoch}/{config.epochs}], Train Loss Ran: "
            f"{train_loss_ran:.8f}, "
            f"Train Loss Ker: {train_loss_ker:.8f}, "
            f"Test Loss Ran: {test_loss_ran:.8f}, "
            f"Test Loss Ker: {test_loss_ker:.8f}, "
        )

        train_loss_list_ran[epoch-1] = train_loss_ran
        test_loss_list_ran[epoch-1] = test_loss_ran

        train_loss_list_ker[epoch-1] = train_loss_ker
        test_loss_list_ker[epoch-1] = test_loss_ker

        plot_losses(
            train_loss_list_ran[1:],
            test_loss_list_ran[1:],
            f"losses/{example}_loss_{model_name}_ran.pdf"
        )
        plot_losses(
            train_loss_list_ker[1:],
            test_loss_list_ker[1:],
            f"losses/{example}_loss_{model_name}_ker.pdf"
        )

    print(
        f"Saving model {model_name}, best epoch range: {best_epoch_ran} "
        f"best epoch on null space: {best_epoch_ker}."
    )
    torch.save(best_model_params_ran, f"models/{example}/{model_name}_ran.pth")
    torch.save(best_model_params_ker, f"models/{example}/{model_name}_ker.pth")

    print("############################################################## ")


def train_single_dp(model_name: str, initial_regul: str, config, example):
    print(f"Start training of {model_name}.")
    if model_name.split("_")[0] == "fbp":
        initrecon = False
    elif model_name.split("_")[0] in ["tv", "ell1"]:
        initrecon = True
    else:
        raise ValueError(f"Unknown model {model_name}.")

    radon_full, radon_limited, radon_null_space = get_radon_operators(example)

    best_test_loss = float('inf')
    best_model_params = None

    model, optimizer, error = get_model(config.training_params)
    TrainDataGen, TestDataGen = get_data_loader(initrecon, initial_regul)

    best_test_loss = float('inf')
    best_model_params = None

    torch.cuda.empty_cache()

    train_loss_list = np.zeros(config.epochs)
    test_loss_list = np.zeros(config.epochs)

    for epoch in np.arange(1, config.epochs+1):
        train_loss = 0.0
        test_loss = 0.0

        model.train()
        for X, Y, Z in TrainDataGen:
            X = X.to(device=config.device, dtype=torch.float)
            Y = Y.to(device=config.device, dtype=torch.float)
            Z = Z.to(device=config.device, dtype=torch.float)

            model_output, X_output = calc_single_dp_output(
                model_name, model, X, Y, Z,
                radon_full, radon_limited, radon_null_space, config
            )

            loss = error(model_output.double(), X_output.double())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.data.item()

        with torch.no_grad():
            model.eval()
            for X, Y, Z in TestDataGen:

                X = X.to(device=config.device, dtype=torch.float)
                Y = Y.to(device=config.device, dtype=torch.float)
                Z = Z.to(device=config.device, dtype=torch.float)

                model_output, X_output = calc_single_dp_output(
                    model_name, model, X, Y, Z,
                    radon_full, radon_limited, radon_null_space, config
                )

                loss = error(model_output.double(), X_output.double())

                test_loss += loss.data.item()

            train_loss = train_loss/config.len_train
            test_loss = test_loss/config.len_test

            if test_loss < best_test_loss:
                best_epoch = epoch
                best_test_loss = test_loss
                best_model_params = model.state_dict().copy()
                torch.save(
                    best_model_params, f"models/{example}/{model_name}.pth"
                )

        print(
            f"Epoch [{epoch}/{config.epochs}], Train Loss: {train_loss:.8f} "
            f"and Test Loss: {test_loss:.8f}"
        )

        train_loss_list[epoch-1] = train_loss
        test_loss_list[epoch-1] = test_loss

        plot_losses(
            train_loss_list[1:],
            test_loss_list[1:],
            f"losses/{example}_loss_{model_name}.pdf"
        )

    print(f"Saving model {model_name}, best epoch: {best_epoch}.")
    torch.save(best_model_params, f"models/{example}/{model_name}.pth")

    print("############################################################## ")


def train_nsn(model_name: str, initial_regul: str, config, example):
    print(f"Start training of {model_name}.")
    if model_name == "fbp_nsn":
        initrecon = False
    elif model_name in ["tv_nsn", "ell1_nsn"]:
        initrecon = True
    else:
        raise ValueError(f"Unknown model {model_name}.")

    radon_full, radon_limited, radon_null_space = get_radon_operators(example)

    best_test_loss = float('inf')
    best_model_params = None

    model, optimizer, error = get_model(config.training_params)
    TrainDataGen, TestDataGen = get_data_loader(initrecon, initial_regul)

    best_test_loss = float('inf')
    best_model_params = None

    torch.cuda.empty_cache()

    train_loss_list = np.zeros(config.epochs)
    test_loss_list = np.zeros(config.epochs)

    for epoch in np.arange(1, config.epochs+1):
        train_loss = 0.0
        test_loss = 0.0

        model.train()
        for X, Y, Z in TrainDataGen:
            X = X.to(device=config.device, dtype=torch.float)
            Y = Y.to(device=config.device, dtype=torch.float)
            Z = Z.to(device=config.device, dtype=torch.float)

            model_output, X_output = calc_nsn_output(
                model_name, model, X, Y, Z,
                radon_full, radon_limited, radon_null_space, config
            )

            loss = error(model_output.double(), X_output.double())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.data.item()

        with torch.no_grad():
            model.eval()
            for X, Y, Z in TestDataGen:

                X = X.to(device=config.device, dtype=torch.float)
                Y = Y.to(device=config.device, dtype=torch.float)
                Z = Z.to(device=config.device, dtype=torch.float)

                model_output, X_output = calc_nsn_output(
                    model_name, model, X, Y, Z,
                    radon_full, radon_limited, radon_null_space, config
                )

                loss = error(model_output.double(), X_output.double())

                test_loss += loss.data.item()

            train_loss = train_loss/config.len_train
            test_loss = test_loss/config.len_test

            if test_loss < best_test_loss:
                best_epoch = epoch
                best_test_loss = test_loss
                best_model_params = model.state_dict().copy()
                torch.save(
                    best_model_params, f"models/{example}/{model_name}.pth"
                )

        print(
            f"Epoch [{epoch}/{config.epochs}], Train Loss: {train_loss:.8f}"
            f" and Test Loss: {test_loss:.8f}"
        )

        train_loss_list[epoch-1] = train_loss
        test_loss_list[epoch-1] = test_loss

        plot_losses(
            train_loss_list[1:],
            test_loss_list[1:],
            f"losses/{example}_loss_{model_name}.pdf"
        )

    print(f"Saving model {model_name}, best epoch: {best_epoch}.")
    torch.save(best_model_params, f"models/{example}/{model_name}.pth")

    print("############################################################## ")


def train_res(model_name: str, initial_regul: str, config, example):
    print(f"Start training of {model_name}.")
    if model_name in ["fbp_res", "fbp_X_res"]:
        initrecon = False
    elif model_name in ["tv_res", "tv_X_res", "ell1_res", "ell1_X_res"]:
        initrecon = True
    else:
        raise ValueError(f"Unknown model {model_name}.")

    if model_name in ["fbp_res", "tv_res", "ell1_res"]:
        radon_full, _, _ = get_radon_operators(example)
    else:
        radon_full = None

    best_test_loss = float('inf')
    best_model_params = None

    TrainDataGen, TestDataGen = get_data_loader(initrecon, initial_regul)
    model, optimizer, error = get_model(config.training_params)

    best_test_loss = float('inf')
    best_model_params = None

    torch.cuda.empty_cache()

    train_loss_list = np.zeros(config.epochs)
    test_loss_list = np.zeros(config.epochs)

    for epoch in np.arange(1, config.epochs+1):
        train_loss = 0.0
        test_loss = 0.0

        model.train()
        for X, Y, Z in TrainDataGen:

            X = X.to(device=config.device, dtype=torch.float)
            Y = Y.to(device=config.device, dtype=torch.float)
            Z = Z.to(device=config.device, dtype=torch.float)

            model_output, X_output = calc_res_output(
                model_name, model, X, Y, radon_full, config
            )

            loss = error(model_output.double(), X_output.double())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.data.item()

        with torch.no_grad():
            model.eval()
            for X, Y, Z in TestDataGen:

                X = X.to(device=config.device, dtype=torch.float)
                Y = Y.to(device=config.device, dtype=torch.float)

                model_output, X_output = calc_res_output(
                    model_name, model, X, Y, radon_full, config
                )

                test_loss += loss.data.item()

            train_loss = train_loss/config.len_train
            test_loss = test_loss/config.len_test

            if test_loss < best_test_loss:
                best_epoch = epoch
                best_test_loss = test_loss
                best_model_params = model.state_dict().copy()
                torch.save(
                    best_model_params, f"models/{example}/{model_name}.pth")

        print(
            f"Epoch [{epoch}/{config.epochs}], Train Loss: {train_loss:.8f}"
            f" and Test Loss: {test_loss:.8f}"
        )

        train_loss_list[epoch-1] = train_loss
        test_loss_list[epoch-1] = test_loss

        plot_losses(
            train_loss_list[1:],
            test_loss_list[1:],
            f"losses/{example}_loss_{model_name}.pdf"
        )

    print(f"Saving model {model_name}, best epoch: {best_epoch}.")
    torch.save(best_model_params, f"models/{example}/{model_name}.pth")

    print("############################################################## ")


"############################################################## "
"############################################################## "
"############################################################## "
"############################################################## "
"############################################################## "
"############################################################## "


def calc_output(
    model_name, model, X, Y, Z,
    radon_full, radon_limited, radon_null_space, config
):
    X = X.to(device=config.device, dtype=torch.float)
    Y = Y.to(device=config.device, dtype=torch.float)
    Z = Z.to(device=config.device, dtype=torch.float)

    if model_name in [
        "fbp_X_res", "tv_X_res", "ell1_X_res", "fbp_res", "tv_res", "ell1_res"
    ]:
        model_output, X_output = calc_res_output(
            model_name, model, X, Y, radon_full, config
        )
    elif model_name in ["fbp_nsn", "tv_nsn", "ell1_nsn"]:
        model_output, X_output = calc_nsn_output(
            model_name, model, X, Y, Z,
            radon_full, radon_limited, radon_null_space, config
        )
    elif model_name in ["fbp_single_dp", "tv_single_dp", "ell1_single_dp"]:
        model_output, X_output = calc_single_dp_output(
            model_name, model, X, Y, Z,
            radon_full, radon_limited, radon_null_space, config
        )
    else:
        raise ValueError(f"Unknown model {model_name}.")
    return model_output, X_output


def train(example: str, model_name: str, initial_regul: str, config):
    print(f"Start training of {model_name}.")
    if model_name.split("_")[0] == "fbp":
        initrecon = False
    elif model_name.split("_")[0] in ["tv", "ell1"]:
        initrecon = True
    else:
        raise ValueError(f"Unknown model {model_name}.")

    radon_full, radon_limited, radon_null_space = get_radon_operators(example)

    best_test_loss = float('inf')
    best_model_params = None

    model, optimizer, error = get_model(config.training_params)
    TrainDataGen, TestDataGen = get_data_loader(initrecon, initial_regul)

    best_test_loss = float('inf')
    best_model_params = None

    torch.cuda.empty_cache()

    train_loss_list = np.zeros(config.epochs)
    test_loss_list = np.zeros(config.epochs)

    for epoch in np.arange(1, config.epochs+1):
        train_loss = 0.0
        test_loss = 0.0

        model.train()
        for X, Y, Z in TrainDataGen:

            model_output, X_output = calc_output(
                model_name,
                model,
                X, Y, Z,
                radon_full,
                radon_limited,
                radon_null_space,
                config
            )

            loss = error(model_output.double(), X_output.double())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.data.item()

        with torch.no_grad():
            model.eval()
            for X, Y, Z in TestDataGen:

                model_output, X_output = calc_output(
                    model_name,
                    model,
                    X, Y, Z,
                    radon_full,
                    radon_limited,
                    radon_null_space,
                    config
                )

                loss = error(model_output.double(), X_output.double())

                test_loss += loss.data.item()

            train_loss = train_loss/config.len_train
            test_loss = test_loss/config.len_test

            if test_loss < best_test_loss:
                best_epoch = epoch
                best_test_loss = test_loss
                best_model_params = model.state_dict().copy()
                torch.save(
                    best_model_params, f"models/{example}/{model_name}.pth"
                )

        print(
            f"Epoch [{epoch}/{config.epochs}], Train Loss: {train_loss:.8f} "
            f"and Test Loss: {test_loss:.8f}"
        )

        train_loss_list[epoch-1] = train_loss
        test_loss_list[epoch-1] = test_loss

        plot_losses(
            train_loss_list[1:],
            test_loss_list[1:],
            f"losses/{example}_loss_{model_name}.pdf"
        )

    print(f"Saving model {model_name}, best epoch: {best_epoch}.")
    torch.save(best_model_params, f"models/{example}/{model_name}.pth")

    print("############################################################## ")


if __name__ == "__main__":
    import argparse

    from src.utils.load_config import load_config
    from src.utils.parser import parse_nested_list

    parser = argparse.ArgumentParser(description="Training inputs.")
    parser.add_argument("example", type=str, help="Which example.")
    parser.add_argument("model_names", type=parse_nested_list, nargs='?')

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
            model_name=model_name[0],
            initial_regul=model_name[1],
            config=config
        )
    print(f"Testing completed for {args.example}.")
