import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import torch  # type: ignore
# import torch_radon
import torch_radon.solvers

from src.models.unet import UNet
from src.utils.radon_operator import filter_sinogram

from .utils import data_prox_func


def calc_nsn_output(
    model_name,
    model,
    X,
    Y,
    Z,
    radon_full,
    radon_limited,
    radon_null_space,
    config
):
    output = model(Y)

    # IF MODEL ON X
    X_output = X
    # X_output = radon_full.forward(X)

    r_output_ns = radon_null_space.forward(output)
    # proj_ns = radon_null_space.backward(filter_sinogram(r_output_ns))
    # model_output = Y + proj_ns

    num_ = X.shape[0]
    model_output = torch.zeros([num_, 1, config.num_angles_full, config.det_count]).to("cuda")
    model_output[:, :, 0:config.num_angles_limited, :] = radon_limited.forward(Y)

    model_output[:, :, config.num_angles_limited:, :] = r_output_ns
    model_output = radon_full.backward(filter_sinogram(model_output))

    return model_output, X_output


def calc_res_output(
    model_name,
    model,
    X,
    Y,
    radon_full,
    config
):
    output = model(Y)

    if model_name in ["fbp_X_res", "tv_X_res", "ell1_X_res", "landweber_X_res"]:
        model_output = Y + output
        X_output = X
    elif model_name in ["fbp_res", "tv_res", "ell1_res"]:
        radon_output = radon_full.forward(output)
        radon_X = radon_full.forward(X)
        radon_Y = radon_full.forward(Y)
        model_output = radon_Y + radon_output
        X_output = radon_X
        model_output = radon_full.backward(filter_sinogram(model_output))
    else:
        raise ValueError(f"Unknown model {model_name}.")

    return model_output, X_output


def calc_single_dp_output(
    model_name,
    model,
    X,
    Y,
    Z,
    radon_full,
    radon_limited,
    radon_null_space,
    config
):
    # X_output = radon_limited.forward(X)
    output = model(Y)

    beta = config.norm
    # if config.factor is not None:
    #     beta *= config.factor
        # beta *= 10

    X_output = X
    # X_output = radon_full.forward(X)

    # model_output = torch.zeros_like(X).to(config.device)

    # r_output_dp = data_prox_func(radon_limited.forward(output), beta)
    # r_output_ns = radon_null_space.forward(output)

    # proj_ran = radon_limited.backward(filter_sinogram(r_output_dp))
    # proj_ns = radon_null_space.backward(filter_sinogram(r_output_ns))
    # model_output = Y + proj_ran + proj_ns

    r_output_ran = radon_limited.forward(output)
    r_output_ns = radon_null_space.forward(output)

    num_ = X.shape[0]
    model_output = torch.zeros([num_, 1, config.num_angles_full, config.det_count]).to("cuda")

    # th = torch.Tensor(np.load("data/data_lodopab/data_processed/train/avg_max.npy")).cuda()
    # UU = r_output_ran
    # UU[abs(UU) > th] = th

    model_output[:, :, 0:config.num_angles_limited, :] = radon_limited.forward(Y) + data_prox_func(r_output_ran, beta)
    model_output[:, :, config.num_angles_limited:, :] = r_output_ns

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(model_output.detach().cpu().numpy().squeeze(), cmap="gray")
    # plt.colorbar()
    # plt.savefig("layer_test_model_output.png")

    model_output = radon_full.backward(filter_sinogram(model_output))

    return model_output, X_output


def calc_onlydp_output(
    model_name,
    model,
    X,
    Y,
    Z,
    radon_full,
    radon_limited,
    radon_null_space,
    config
):
    output = model(Y)
    beta = config.norm
    X_output = X

    # res = radon_full.backward(
    #     data_prox_func(
    #         radon_full.forward(output), beta
    #     )
    # )
    th = int(model_name.split("_")[2])

    beta = config.norm  # ca. 165.6752
    res_ran = radon_limited.forward(output)
    res_ran[torch.abs(res_ran) > th] = th
    res_nsn = radon_null_space.forward(output)
    res = torch.zeros([X.shape[0], 1, config.num_angles_full, config.det_count]).to(config.device)
    res[:, :, 0:config.num_angles_limited, :] = res_ran
    res[:, :, config.num_angles_limited:, :] = res_nsn
    model_output = Y + radon_full.backward(filter_sinogram(res))
    # model_output = Y + radon_limited.backward(filter_sinogram(data_prox_func(radon_limited.forward(output), beta)))

    return model_output, X_output


if __name__ == "__main__":
    print("Start")
    import numpy as np

    from src.utils.load_config import load_config
    from src.utils.radon_operator import get_radon_operators

    example = "synthetic"
    config = load_config(example)
    radon_full, radon_limited, radon_null_space = get_radon_operators(example)

    # radon_limited = radon_full

    x = np.linspace(-config.N//2, config.N//2-1, config.N)
    X, Y = np.meshgrid(x, x)
    image = np.zeros([config.N, config.N])
    # image[50:80, 80:100] = 1
    # image[5:25, 5:25] = 1
    # image[(X)**2 + (Y+5)**2 <= 8**2] = 0.5

    from PIL import Image

    image = Image.open("ncat.png").convert("L").resize((128, 128))
    image = np.array(image).astype(np.float64)
    image /= np.max(np.abs(image))

    X = torch.Tensor(image).to(config.device)
    Z = radon_limited.forward(X)
    Y = radon_limited.backward(filter_sinogram(Z))
    # Y = X

    plt.figure()
    plt.imshow(X.cpu().numpy().squeeze(), cmap="gray")
    plt.savefig("layer_test_X.png")

    plt.figure()
    plt.imshow(Y.cpu().numpy().squeeze(), cmap="gray")
    plt.savefig("layer_test_Y.png")

    plt.figure()
    plt.imshow(Z.cpu().numpy().squeeze(), cmap="gray")
    plt.savefig("layer_test_Z.png")

    r_output_ns = radon_null_space.forward(Y)
    proj_ns = radon_null_space.backward(filter_sinogram(r_output_ns))
    # model_output = Y + proj_ns

    num_ = X.shape[0]
    model_output = torch.zeros(
        [1, 1, config.num_angles_full, config.det_count]).to("cuda")
    model_output[:, :, 0:config.num_angles_limited, :] = radon_limited.forward(Y)
    model_output[:, :, config.num_angles_limited:, :] = r_output_ns

    model_output = radon_full.backward(filter_sinogram(model_output))

    # model_output = radon_limited.backward(filter_sinogram(radon_limited.forward(Y)))

    plt.figure()
    plt.imshow(model_output.cpu().numpy().squeeze(), cmap="gray")
    plt.colorbar()
    plt.savefig("layer_test_output_limited.png")

    # r_output_ns = radon_null_space.forward(Y)
    # proj_ns = radon_null_space.backward(filter_sinogram(r_output_ns))
    # model_output = Y + proj_ns

    num_ = X.shape[0]
    model_output_full = torch.zeros(
        [1, 1, config.num_angles_full, config.det_count]).to("cuda")
    model_output_full[:, :, 0:config.num_angles_limited, :] = radon_full.forward(Y)[:, :, 0:config.num_angles_limited, :]
    model_output_full[:, :, config.num_angles_limited:, :] = radon_full.forward(Y)[:, :, config.num_angles_limited:, :]

    model_output_full = radon_full.backward(filter_sinogram(model_output_full))

    plt.figure()
    plt.imshow(model_output_full.cpu().numpy().squeeze(), cmap="gray")
    plt.colorbar()
    plt.savefig("layer_test_output_full.png")

    expected_out = radon_limited.backward(
        filter_sinogram(radon_limited.forward(Y))
    )

    plt.figure()
    plt.imshow(expected_out.cpu().numpy().squeeze(), cmap="gray")
    plt.colorbar()
    plt.savefig("layer_test_expected_out.png")

    sum_out = radon_limited.backward(
        filter_sinogram(radon_limited.forward(X))
    ) + radon_null_space.backward(filter_sinogram(radon_null_space.forward(X)))

    plt.figure()
    plt.imshow(sum_out.cpu().numpy().squeeze(), cmap="gray")
    plt.colorbar()
    plt.savefig("layer_test_sum_out.png")

    Y_comb = torch.zeros([1, 1, 180, 200]).to("cuda")
    Y_comb[:, :, 0:config.num_angles_limited, :] = radon_limited.forward(Y)
    Y_comb[:, :, config.num_angles_limited:, :] = radon_null_space.forward(X)
    comb_out = radon_full.backward(filter_sinogram(Y_comb))

    plt.figure()
    plt.imshow(comb_out.cpu().numpy().squeeze(), cmap="gray")
    plt.colorbar()
    plt.savefig("layer_test_comb_out.png")

    print("Ende")
