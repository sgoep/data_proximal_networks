import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_radon

# from example import example
from src.algorithms.ell1_shearlet import ell1_shearlet
from src.algorithms.ell1_wavelet import ell1_wavelet
from src.algorithms.my_shearlet import ShearletTransform
from src.algorithms.total_variation import tv
from src.data.data_loader import DataLoader
from src.data.utils import linear_regression
from src.utils.load_config import load_config
from src.utils.load_data import read_h5_file
from src.utils.radon_operator import filter_sinogram, get_radon_operators
from src.utils.test_utils import print_errors
from src.visualization.visualization import visualization_with_zoom


def test_init_recon(example):
    config = load_config(example)
    print("Start.")
    radon_full, radon_limited, _ = get_radon_operators(example)

    index = 1366
    # index = 1

    if example == "synthetic":
        D = DataLoader(
            np.arange(0, 1500),
            initrecon=True,
            example=example,
            initial_regul="tv",
            test_train=None,
        )
        X, Y, Z = D[index]  # type: ignore

        X = torch.Tensor(X.squeeze()).to("cuda")
        sinogram = torch.Tensor(Z.squeeze()).to("cuda")

        torch.manual_seed(1)

        sinogram = radon_limited.forward(X)
        noise = 0.03 * torch.max(torch.abs(sinogram)) * torch.randn(*sinogram.shape).to(config.device)
        sinogram += noise
        # sinogram = extension_with_zero(sinogram)
    else:
        # D = DataLoader(
        #     np.arange(0, 1500),
        #     initrecon=True,
        #     example=example,
        #     initial_regul="ell1",
        #     test_train="validation"
        # )
        # X, Y, Z = D[index]  # type: ignore
        # X = torch.Tensor(X.squeeze()).to("cuda")
        # sinogram = torch.Tensor(Z.squeeze()).to("cuda")

        idx = 000
        file_name_data = f"data/data_lodopab/data_original/observations/train/observation_train_000.hdf5"

        data = read_h5_file(file_name_data)[0, :, :]  # * 1000
        data = data[0 : config.num_angles_limited, :]
        file_name_data = f"data/data_lodopab/data_original/ground_truth/train/ground_truth_train_000.hdf5"

        X = read_h5_file(file_name_data)[0, :, :]
        X = X.T

        plt.figure()
        plt.imshow(data, cmap="gray")
        plt.colorbar()
        plt.axis("off")
        plt.savefig("init_recon_data.png", bbox_inches="tight")

        plt.figure()
        plt.imshow(X, cmap="gray")
        plt.colorbar()
        plt.axis("off")
        plt.savefig("init_recon_orig.png", bbox_inches="tight")

        X = torch.Tensor(X.squeeze()).to("cuda")

        sinogram_tensor = torch.Tensor(data).to("cuda")
        a = linear_regression(radon_limited.forward(X), sinogram_tensor, lr=0.0001, num_iterations=2000)

        sinogram = a * sinogram_tensor  # + b
        # sinogram = sinogram_tensor

        plt.figure()
        plt.imshow(sinogram.cpu().numpy(), cmap="gray")
        plt.colorbar()
        plt.axis("off")
        plt.savefig("init_recon_scaled_data.png", bbox_inches="tight")

        plt.figure()
        plt.imshow(
            radon_limited.backward(filter_sinogram(radon_limited.forward(X))).cpu().numpy().squeeze(),
            cmap="gray",
        )
        plt.colorbar()
        plt.axis("off")
        plt.savefig("init_recon_X_data.png", bbox_inches="tight")

    x0 = torch.zeros([config.N, config.N]).to("cuda")

    # p_0 = 0.1
    # p_1 = 0.8
    # Niter = 200

    p_0 = 0.0001
    p_1 = 0.0001
    Niter = 100
    from pytorch_wavelets import DWTForward, DWTInverse

    wavelet = DWTForward(J=3, mode="zero", wave="db3").cuda()  # Accepts all wave types available to PyWavelets
    iwavelet = DWTInverse(mode="zero", wave="db3").cuda()

    # f = ell1_wavelet(
    #     wavelet=wavelet,
    #     iwavelet=iwavelet,
    #     A=radon_limited,
    #     sinogram=sinogram,
    #     p_0=p_0,
    #     p_1=p_1,
    #     Niter=Niter,
    #     ground_truth=X,
    #     print_flag=True
    # )

    L = 500
    Niter = 1000
    alpha = 0.5
    tau = 1 / L
    sigma = 1 / L
    # tau = 0.001
    # sigma = 0.001
    theta = 1
    f = tv(
        x0=x0,
        A=radon_limited,
        g=sinogram,
        alpha=alpha,
        tau=tau,
        sigma=sigma,
        theta=theta,
        L=L,
        Niter=Niter,
        ground_truth=X,
        print_flag=True,
    )

    # shearlet = ShearletTransform(config.N, config.N, [0.5]*5)

    # f = ell1_shearlet(
    #     shearlet=shearlet,
    #     A=radon_limited,
    #     sinogram=sinogram,
    #     p_0=p_0,
    #     p_1=p_1,
    #     Niter=Niter,
    #     ground_truth=X,
    #     print_flag=True
    # )
    fbp = radon_limited.backward(filter_sinogram(sinogram))

    visualization_with_zoom(example, f.cpu().numpy(), False, True, "init_recon_rec.png")

    print("Finished.")


if __name__ == "__main__":
    test_init_recon("lodopab")
