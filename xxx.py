import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_radon

from src.algorithms.ell1_shearlet import ell1_shearlet
from src.algorithms.my_shearlet import ShearletTransform
from src.algorithms.total_variation import tv
from src.data.data_loader import DataLoader
from src.data.utils import linear_regression
from src.utils.load_config import load_config
from src.utils.load_data import read_h5_file
from src.utils.radon_operator import filter_sinogram, get_radon_operators
from src.utils.test_utils import print_errors


def test_landweber(example):
    config = load_config(example)
    print("Start.")
    radon_full, radon_limited, radon_null_space = get_radon_operators(example)

    index = 1366

    print(config.det_count)

    if example == "synthetic":
        D = DataLoader(
            np.arange(0, 1500),
            initrecon=False,
            example=example,
            initial_regul="tv",
            test_train=None,
        )
        X, Y, Z = D[index]  # type: ignore

        X = torch.Tensor(X.squeeze()).to("cuda")
        sinogram = torch.Tensor(Z.squeeze()).to("cuda")

        torch.manual_seed(1)
        sinogram = radon_full.forward(X)
        noise = (
            0.0
            * torch.max(torch.abs(sinogram))
            * torch.randn(*sinogram.shape).to(config.device)
        )
        sinogram += noise

        sinogram_ran = radon_limited.forward(X)
        noise = (
            0.03
            * torch.max(torch.abs(sinogram_ran))
            * torch.randn(*sinogram_ran.shape).to(config.device)
        )
        sinogram_ran += noise

        sinogram_nsn = radon_null_space.forward(X)
        noise = (
            0.0
            * torch.max(torch.abs(sinogram_nsn))
            * torch.randn(*sinogram_nsn.shape).to(config.device)
        )
        sinogram_nsn += noise
        sinogram = sinogram_ran
        # sinogram = extension_with_zero(sinogram)
    else:
        D = DataLoader(
            np.arange(0, 1500),
            initrecon=False,
            example=example,
            initial_regul="fbp",
            test_train="validation",
        )
        X, Y, Z = D[index]  # type: ignore

        X = torch.Tensor(X.squeeze()).to("cuda")
        sinogram = torch.Tensor(Z.squeeze()).to("cuda")  # /1000

        a, b = linear_regression(
            radon_limited.forward(X), sinogram, lr=0.0001, num_iterations=200
        )
        print(a)
        print(b)
        sinogram = a * sinogram

    # X = np.zeros([config.N, config.N])
    # X[5:15, 5:15] = 1
    # X = torch.Tensor(X.squeeze()).to("cuda")
    # Y = X
    # sinogram = radon_full.forward(Y)
    # noise = 0.*torch.max(
    #             torch.abs(sinogram)
    #         )*torch.randn(*sinogram.shape).to(config.device)
    # sinogram += noise

    # sinogram_ran = radon_limited.forward(Y)
    # noise = 0.*torch.max(
    #             torch.abs(sinogram_ran)
    #         )*torch.randn(*sinogram_ran.shape).to(config.device)
    # sinogram_ran += noise

    # sinogram_nsn = radon_null_space.forward(Y)
    # noise = 0.0*torch.max(
    #             torch.abs(sinogram_nsn)
    #         )*torch.randn(*sinogram_nsn.shape).to(config.device)
    # sinogram_nsn += noise

    print(sinogram.shape)

    # fbp = radon_limited.backward(filter_sinogram(sinogram))
    # Y = torch.Tensor(Y.squeeze()).to("cuda")
    # sinogram = radon.forward(Y)

    x0 = torch.zeros([config.N, config.N]).to("cuda")

    # # fbp = radon_limited.backward(filter_sinogram(sinogram))
    # landweber_full = torch_radon.solvers.Landweber(
    #     radon_full, projection=None, grad=False
    # )
    landweber_limited = torch_radon.solvers.Landweber(
        radon_limited, projection=None, grad=False
    )
    # landweber_nsn = torch_radon.solvers.Landweber(
    #     radon_null_space, projection=None, grad=False
    # )

    # rate = 1e-4
    # print(sinogram.shape)
    # f_rec = landweber_limited.run(x0, sinogram, rate, iterations=1000)

    rate = 1e-6
    # print(sinogram.shape)
    f_rec = landweber_limited.run(x0, sinogram, rate, iterations=1000)

    plt.figure()
    plt.imshow(f_rec.cpu().numpy().squeeze())  # , cmap="gray")
    plt.colorbar()
    plt.savefig("xxx_landweber_rec.png")

    L = 500
    Niter = 1000
    alpha = 0.2
    tau = 1 / L
    sigma = 1 / L
    theta = 1
    f_tv = tv(
        x0,
        radon_limited,
        sinogram,
        alpha,
        tau,
        sigma,
        theta,
        L,
        Niter,
        ground_truth=X,
        print_flag=True,
    )

    plt.figure()
    plt.imshow(f_tv.cpu().numpy().squeeze())  # , cmap="gray")
    plt.colorbar()
    plt.savefig("xxx_tv_rec.png")

    plt.figure()
    plt.imshow(X.cpu().numpy().squeeze())  # , cmap="gray")
    plt.colorbar()
    plt.savefig("xxx_gt.png")
    # f_full = landweber_full.run(x0, sinogram, rate, iterations=100)
    # f_ran = landweber_ran.run(x0, sinogram_ran, rate, iterations=100)
    # f_nsn = landweber_nsn.run(x0, sinogram_nsn, rate, iterations=100)

    # f_fbp_added = radon_limited.backward(filter_sinogram(sinogram_ran)) + radon_null_space.backward(filter_sinogram(sinogram_nsn))

    # sino_full = torch.zeros_like(sinogram).to("cuda")
    # sino_full[0:config.num_angles_limited, :] = sinogram_ran
    # sino_full[config.num_angles_limited:, :] = sinogram_nsn

    # f_fbp_combined = radon_full.backward(filter_sinogram(sino_full))

    # f_fbp_full = radon_full.backward(filter_sinogram(sinogram))

    # sino_limited = radon_limited.forward(X)

    # fbp_limited = radon_limited.backward(filter_sinogram(sino_limited))

    # forward_full_fbp_limited = radon_full.forward(fbp_limited)

    # landweber_sino_limited = landweber_ran.run(x0, sino_limited, rate, iterations=1000)
    # forward_landweber = radon_full.forward(landweber_sino_limited)

    # plt.figure()
    # plt.imshow(radon_full.forward(X).cpu().numpy().squeeze()) #, cmap="gray")
    # plt.colorbar()
    # plt.savefig("init_sino_full.png")

    # plt.figure()
    # plt.imshow(forward_full_fbp_limited.cpu().numpy().squeeze()) #, cmap="gray")
    # plt.colorbar()
    # plt.savefig("init_forward_fbp_full.png")

    # plt.figure()
    # plt.imshow(forward_landweber.cpu().numpy().squeeze()) #, cmap="gray")
    # plt.colorbar()
    # plt.savefig("init_forward_landweber.png")

    # plt.figure()
    # plt.imshow(f_full.cpu().numpy(), cmap="gray")
    # plt.colorbar()
    # plt.savefig("init_landweber_full.png")

    # plt.figure()
    # plt.imshow(f_ran.cpu().numpy(), cmap="gray")
    # plt.colorbar()
    # plt.savefig("init_landweber_ran.png")

    # plt.figure()
    # plt.imshow(f_nsn.cpu().numpy(), cmap="gray")
    # plt.colorbar()
    # plt.savefig("init_landweber_nsn.png")

    # plt.figure()
    # plt.imshow((f_ran + f_nsn).cpu().numpy(), cmap="gray")
    # plt.colorbar()
    # plt.savefig("init_landweber_added.png")

    # plt.figure()
    # plt.imshow(f_fbp_added.cpu().numpy().squeeze(), cmap="gray")
    # plt.colorbar()
    # plt.savefig("init_fbp_added.png")

    # plt.figure()
    # plt.imshow(f_fbp_combined.cpu().numpy().squeeze(), cmap="gray")
    # plt.colorbar()
    # plt.savefig("init_fbp_combined.png")

    # plt.figure()
    # plt.imshow(f_fbp_full.cpu().numpy().squeeze(), cmap="gray")
    # plt.colorbar()
    # plt.savefig("init_fbp_full.png")

    # plt.figure()
    # plt.imshow(Y[0, :, :], cmap="gray")
    # plt.colorbar()
    # plt.savefig("init_recon_init.png")

    # plt.figure()
    # plt.imshow(radon_full.forward(f).cpu().numpy())#, cmap="gray")
    # plt.colorbar()
    # plt.savefig("init_recon_radon.png")

    # limited_angle_data = radon_full.forward(X)
    # limited_angle_data[config.num_angles_limited:, :] = 0

    # N = 128
    # angles_full = np.linspace(0, np.pi, 180, endpoint=False)
    # angles_limited = angles_full[angles_full <= 3*np.pi/4]
    # det_count = 200
    # det_spacing = 1
    # clip_to_circle = False

    # radon_full = torch_radon.Radon(
    #         resolution=N,
    #         angles=angles_full,
    #         det_count=det_count,
    #         det_spacing=det_spacing,
    #         clip_to_circle=clip_to_circle)

    # radon_limited = torch_radon.Radon(
    #         resolution=N,
    #         angles=angles_limited,
    #         det_count=det_count,
    #         det_spacing=det_spacing,
    #         clip_to_circle=clip_to_circle)

    # data_full = radon_full.forward(X)
    # data_limited = radon_limited.forward(X)

    # fbp_full = radon_full.backward(filter_sinogram(data_full))
    # fbp_limited = radon_limited.backward(filter_sinogram(data_limited))

    # data_fbp_full = radon_full.forward(fbp_full)
    # data_fbp_limited = radon_limited.forward(fbp_limited)

    # limited_angle_data = radon_limited.forward(X)

    # plt.figure()
    # plt.imshow(limited_angle_data.cpu().numpy().squeeze())
    # plt.colorbar()
    # plt.savefig("xxx_limited_angle_data.png")

    # fbp_limited_angle_data = radon_limited.backward(filter_sinogram(limited_angle_data))

    # plt.figure()
    # plt.imshow(fbp_limited_angle_data.cpu().numpy().squeeze())
    # plt.colorbar()
    # plt.savefig("xxx_fbp_limited_angle_data.png")

    # forward_fbp_limited_angle_data = radon_limited.forward(fbp_limited_angle_data)

    # rec = radon.backward(filter_sinogram(sinogram)).cpu().numpy().squeeze()
    # plt.figure()
    # plt.imshow(forward_fbp_limited_angle_data.cpu().numpy().squeeze())
    # plt.colorbar()
    # plt.savefig("xxx_recon.png")

    # print(forward_fbp_limited_angle_data.shape)
    # print(limited_angle_data[0:config.num_angles_limited, :].shape)
    # plt.figure()
    # plt.imshow((limited_angle_data[0:config.num_angles_limited, :]-forward_fbp_limited_angle_data.squeeze()).cpu().numpy().squeeze())
    # plt.colorbar()
    # plt.savefig("xxx_diff_data_2.png")

    # plt.figure()
    # plt.imshow(sinogram.cpu().numpy(), cmap="gray")
    # plt.colorbar()
    # plt.savefig("init_recon_sino.png")

    # # fbp = fbp.cpu().numpy().squeeze().astype(np.float32)
    # f = f.cpu().numpy().squeeze().astype(np.float32)
    # print_errors(rec, X.cpu().numpy().squeeze().astype(np.float32))
    # print_errors(f, X.cpu().numpy().squeeze().astype(np.float32))

    print("Finished.")


if __name__ == "__main__":
    test_landweber("lodopab")
