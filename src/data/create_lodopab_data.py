from os import listdir
from os.path import isfile, join

import numpy as np
import torch
import torch_radon
from pytorch_wavelets import DWTForward, DWTInverse
from scipy import stats

# from config import config
from src.algorithms.ell1_wavelet import ell1_wavelet
from src.algorithms.total_variation import tv
from src.utils.load_config import load_config
from src.utils.load_data import read_h5_file
from src.utils.radon_operator import filter_sinogram, get_radon_operators

MAX_TRAIN = 1500
MAX_VALIDATION = 1500
MAX_TEST = 300

L = [
    ("train", MAX_TRAIN),
    ("test", MAX_TEST),
    ("validation", MAX_VALIDATION),
]


def create_data_lodopab():
    """
    Generates and processes training, validation, and test data for the LoDoPaB dataset.

    This function:
    - Loads configuration settings for the dataset.
    - Applies the Radon transform to ground truth images.
    - Computes limited-angle sinograms with estimated noise.
    - Performs multiple reconstruction methods (FBP, Landweber, TV, ℓ₁-wavelet).
    - Saves the processed data for further analysis.

    The reconstructions include:
    - **Filtered Back Projection (FBP)**
    - **Landweber Iteration**
    - **Total Variation (TV) Minimization**
    - **ℓ₁-Regularized Wavelet Reconstruction**

    Args:
        None

    Returns:
        None. Saves processed data as `.npy` files in `data/data_lodopab/data_processed/`.
    """

    print("Start data creation.")

    config = load_config("lodopab")
    _, radon_limited, _ = get_radon_operators("lodopab")

    norm, avg_max = 0.0, 0.0

    wavelet = DWTForward(J=5, mode="zero", wave="db3").cuda()
    iwavelet = DWTInverse(mode="zero", wave="db3").cuda()

    for which, max_num in L:
        print(f"Now {which}, {max_num}.")
        # zip(
        # ["train", "test", "validation"], [MAX_TRAIN, MAX_TEST, MAX_VALIDATION]
        # ):
        file_names = [
            f.split(f"{which}_")[1].split(".hdf5")[0]
            for f in listdir(f"data/data_lodopab/data_original/ground_truth/{which}")
            if isfile(join(f"data/data_lodopab/data_original/ground_truth/{which}", f))
        ]

        num_files = int(np.ceil(max_num / 128))
        landweber_limited = torch_radon.solvers.Landweber(
            radon_limited, projection=None, grad=False
        )
        counter = 0

        for filename in file_names[:num_files]:
            ground_truths = read_h5_file(
                f"data/data_lodopab/data_original/ground_truth/{which}/ground_truth_{which}_{filename}.hdf5"
            )
            observations = read_h5_file(
                f"data/data_lodopab/data_original/observations/{which}/observation_{which}_{filename}.hdf5"
            )

            for i in range(ground_truths.shape[0]):
                gt = np.rot90(ground_truths[i, :, :])
                gt = gt / np.max(np.abs(gt))
                radon_gt = radon_limited.forward(torch.Tensor(gt.copy()).cuda())

                sinogram = observations[i, 0 : config.num_angles_limited, :]
                y0 = radon_gt.cpu().numpy()
                slope, intercept, r, p, std_err = stats.linregress(
                    sinogram.flatten(), y0.flatten()
                )
                scaled_data = slope * sinogram + intercept
                sinogram_tensor = torch.Tensor(scaled_data).cuda()

                estimated_noise = sinogram_tensor.cpu().numpy() - radon_gt.cpu().numpy()
                norm += np.linalg.norm(estimated_noise) / max_num
                avg_max += np.max(np.abs(estimated_noise)) / max_num

                fbp_rec = (
                    radon_limited.backward(filter_sinogram(sinogram_tensor))
                    .cpu()
                    .numpy()
                )

                landweber_rec = (
                    landweber_limited.run(
                        torch.zeros([config.N, config.N]).cuda(),
                        sinogram_tensor,
                        1e-6,
                        iterations=1000,
                    )
                    .cpu()
                    .numpy()
                )

                f_tv = (
                    tv(
                        torch.zeros([config.N, config.N]).cuda(),
                        radon_limited,
                        sinogram_tensor,
                        alpha=0.5,
                        tau=1 / 500,
                        sigma=1 / 500,
                        theta=1,
                        L=500,
                        Niter=2000,
                        ground_truth=None,
                        print_flag=False,
                    )
                    .cpu()
                    .numpy()
                )

                f_ell1 = (
                    ell1_wavelet(
                        wavelet,
                        iwavelet,
                        radon_limited,
                        sinogram_tensor,
                        p_0=0.00001,
                        p_1=0.001,
                        Niter=100,
                        ground_truth=None,
                        print_flag=False,
                    )
                    .cpu()
                    .numpy()
                )

                base_path = f"data/data_lodopab/data_processed/{which}/single_files"
                np.save(f"{base_path}/phantom_{counter}.npy", gt)
                np.save(f"{base_path}/init_regul_tv_{counter}.npy", f_tv)
                np.save(f"{base_path}/init_regul_ell1_{counter}.npy", f_ell1)
                np.save(f"{base_path}/landweber_{counter}.npy", landweber_rec)
                np.save(f"{base_path}/fbp_{counter}.npy", fbp_rec)
                np.save(
                    f"{base_path}/limited_angle_data_{counter}.npy",
                    sinogram_tensor.cpu().numpy().squeeze(),
                )

                counter += 1
                if (np.mod(counter, 100) == 0) or (counter == max_num):
                    print(f"{which}: {counter}/{max_num}")
                if counter == max_num:
                    break

        np.save(f"data/data_lodopab/data_processed/{which}/norm.npy", norm)
        np.save(f"data/data_lodopab/data_processed/{which}/avg_max.npy", avg_max)

    print("Finished.")


if __name__ == "__main__":
    create_data_lodopab()
