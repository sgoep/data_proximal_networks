import h5py  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore
import torch_radon

# from config import config
from src.algorithms.ell1_shearlet import ell1_shearlet
from src.algorithms.my_shearlet import ShearletTransform
from src.algorithms.total_variation import tv
from src.utils.load_config import load_config
from src.utils.radon_operator import filter_sinogram, get_radon_operators


def create_data_synthetic():
    """
    Generates and processes synthetic Shepp-Logan phantom data for reconstruction experiments.

    This function:
    - Loads configuration settings for the synthetic dataset.
    - Applies the Radon transform to synthetic phantom images.
    - Adds Gaussian noise to the sinograms to simulate real-world conditions.
    - Performs multiple reconstruction methods:
        - **Filtered Back Projection (FBP)**
        - **Landweber Iteration**
        - **Total Variation (TV) Minimization**
        - **ℓ₁-Regularized Shearlet Reconstruction**
    - Saves the processed data for further analysis.

    The dataset consists of 1500 synthetic images that undergo forward projection and reconstruction.

    Args:
        None

    Returns:
        None. Saves processed data as `.npy` files in `data/data_synthetic/`.
    """
    config = load_config("synthetic")
    _, radon_limited, _ = get_radon_operators("synthetic")

    images = h5py.File("data/data_synthetic/randshepp.mat")["data"][:].transpose(
        [-1, 0, 1]
    )

    shearlet = ShearletTransform(config.N, config.N, [0.5] * 3)

    Num = 1500
    N = config.N
    phantom_all = np.zeros([Num, N, N])
    fbp_all = np.zeros([Num, N, N])
    landweber_all = np.zeros([Num, N, N])
    init_regul_tv = np.zeros([Num, N, N])
    init_regul_ell1 = np.zeros([Num, N, N])
    limited_angle_data = np.zeros([Num, config.num_angles_limited, config.det_count])
    landweber_limited = torch_radon.solvers.Landweber(
        radon_limited, projection=None, grad=False
    )
    norm, avg_max = 0.0, 0.0
    for index in range(Num):
        phantom = images[index, :, :] / np.max(images[index, :, :])
        phantom_all[index, :, :] = phantom
        sinogram = radon_limited.forward(torch.Tensor(phantom).cuda())

        noise = (
            0.03 * torch.max(torch.abs(sinogram)) * torch.randn(*sinogram.shape).cuda()
        )
        norm += np.linalg.norm(noise.cpu().numpy())
        avg_max += np.max(np.abs(noise.cpu().numpy()))

        sinogram += noise
        filtered_sino = filter_sinogram(sinogram)
        fbp_all[index, :, :] = radon_limited.backprojection(filtered_sino).cpu().numpy()

        landweber_all[index, :, :] = (
            landweber_limited.run(
                torch.zeros([config.N, config.N]).cuda(),
                sinogram,
                1e-4,
                iterations=1000,
            )
            .cpu()
            .numpy()
        )

        x0 = torch.zeros([128, 128]).to("cuda")
        L = 500
        Niter = 1000
        alpha = 0.05
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
            ground_truth=None,
            print_flag=False,
        )

        p_0 = 0.0005
        p_1 = 0.05
        Niter = 100

        f_ell1 = ell1_shearlet(
            shearlet=shearlet,
            A=radon_limited,
            sinogram=sinogram,
            p_0=p_0,
            p_1=p_1,
            Niter=Niter,
            ground_truth=None,
            print_flag=False,
        )

        if (index + 1) % 100 == 0:
            print(f"{index + 1}/1500")

        init_regul_tv[index, :, :] = f_tv.cpu().numpy()
        init_regul_ell1[index, :, :] = f_ell1.cpu().numpy()

        if np.mod(index + 1, 100) == 0:
            print(f"{index+1}/{Num}")

    np.save("data/data_synthetic/avg_max", avg_max / Num)
    np.save("data/data_synthetic/norm", norm / Num)
    np.save("data/data_synthetic/phantom", phantom_all)
    np.save("data/data_synthetic/fbp", fbp_all)
    np.save("data/data_synthetic/landweber", landweber_all)
    np.save("data/data_synthetic/init_regul_tv", init_regul_tv)
    np.save("data/data_synthetic/init_regul_ell1", init_regul_ell1)
    np.save("data/data_synthetic/limited_angle_data", limited_angle_data)
    # np.save("data/data_synthetic/noise_array", noise_array / N)

    print("Finished.")


if __name__ == "__main__":
    create_data_synthetic()
