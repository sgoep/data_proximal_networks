# %%
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch_radon
from pytorch_wavelets import DWTForward, DWTInverse
from scipy.ndimage import gaussian_filter
from skimage.draw import ellipse
from skimage.metrics import mean_squared_error as mse

from src.algorithms.ell1_shearlet import ell1_shearlet
from src.algorithms.ell1_wavelet import ell1_wavelet
from src.algorithms.my_shearlet import ShearletTransform
from src.algorithms.total_variation import tv
from src.utils.load_config import load_config
from src.utils.radon_operator import (
    filter_sinogram,
    get_radon_operator,
    get_radon_operators,
)


def get_ellipses(N):
    random.seed(1)
    image = np.zeros((N, N))  # Initialize with a black background
    # Add multiple ellipses with random positions, sizes, orientations, and transparency
    num_ellipses = 30
    center_point = N // 2  # Center of the image

    for _ in range(num_ellipses):
        # Random size
        width = random.randint(20, 25)
        height = random.randint(10, 15)

        # Random center, clustered around the middle of the array
        spread = N // 3  # Limit how far the ellipses can be from the center
        center_x = random.randint(center_point - spread, center_point + spread)
        center_y = random.randint(center_point - spread, center_point + spread)

        # Generate ellipse, ensuring it doesn't go out of bounds
        rr, cc = ellipse(center_x, center_y, width // 2, height // 2, shape=image.shape)

        # Random intensity for the ellipse
        intensity = random.uniform(0.2, 1.0)

        # Add the ellipse to the image
        image[rr, cc] = intensity
    return image / np.max(image)


def generate_random_ellipses_image_grayscale(height, width, num_ellipses):
    # Create a black canvas with an alpha channel (2 channels: Gray, Alpha)
    image = np.zeros((height, width), dtype=np.uint8)

    np.random.seed(1337)
    for _ in range(num_ellipses):
        # Random center, axes lengths, angle for the ellipse
        center = (np.random.randint(0, width), np.random.randint(0, height))
        axes = (np.random.randint(10, width // 4), np.random.randint(10, height // 4))
        angle = np.random.randint(0, 180)

        # Random intensity (gray value) and alpha for transparency
        gray_value = np.random.randint(50, 256)  # Gray value (intensity)
        alpha_value = np.random.randint(50, 256)  # Alpha (50-255 for transp.)
        color = (gray_value, alpha_value)

        # Create an ellipse mask
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(mask, center, axes, angle, 0, 360, color, -1)

        # Combine the mask with the image using alpha blending
        alpha_blend = mask / 255.0
        image = (1 - alpha_blend) * image + alpha_blend * mask

    return image / np.max(image)


def create_lotus_example():

    f = scipy.io.loadmat("data/LotusData256.mat")

    data = f["m"].T

    wavelet = DWTForward(J=5, mode="zero", wave="db3").cuda()
    iwavelet = DWTInverse(mode="zero", wave="db3").cuda()

    angles = np.linspace(0, 360, data.shape[0], endpoint=True)
    angles = angles * np.pi / 180

    angles = angles[angles <= 3 * np.pi / 4]

    source_distance = 2240 * 540 / 120
    fan_sensor_spacing = 540 / 630
    num_detectors = data.shape[1]

    N = 256

    data = torch.Tensor(data).cuda()
    data = data[0 : len(angles), :]

    A = torch_radon.RadonFanbeam(
        N,
        angles,
        source_distance,
        det_count=num_detectors,
        det_spacing=fan_sensor_spacing,
        clip_to_circle=False,
    )

    rec_fbp = A.backward(filter_sinogram(data))

    landweber = torch_radon.solvers.Landweber(A, projection=None, grad=False)

    x0 = torch.zeros([N, N]).cuda()

    rec_land = landweber.run(x0, data, 1e-6, iterations=10000)

    L = 500
    Niter = 1000
    alpha = 0.005
    tau = 1 / L
    sigma = 1 / L
    theta = 1
    f_tv = tv(
        x0,
        A,
        data,
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

    wavelet = DWTForward(J=5, mode="zero", wave="db3").cuda()
    iwavelet = DWTInverse(mode="zero", wave="db3").cuda()

    f_ell1 = ell1_wavelet(
        wavelet,
        iwavelet,
        A,
        data,
        p_0=0.00009,
        p_1=0.00001,
        Niter=500,
        ground_truth=None,
        print_flag=False,
    )
    return (rec_fbp, data, rec_fbp, f_tv, f_ell1, rec_land)


def create_ellipse_example(example):

    config = load_config(example)
    _, radon_limited, _ = get_radon_operators(example)

    image = generate_random_ellipses_image_grayscale(
        height=config.N,
        width=config.N,
        num_ellipses=6,
    )

    N = config.N

    from PIL import Image

    image = Image.open("ncat.png").convert("L").resize((config.N, config.N))
    image = np.array(image).astype(np.float64)
    image /= np.max(np.abs(image))
    # print(image.shape)

    image = torch.Tensor(image).to(config.device)

    # data_full = radon_full.forward(image)

    data_limited = radon_limited.forward(image)

    noise = (
        0.01
        * torch.max(torch.abs(data_limited))
        * torch.randn(*data_limited.shape).to(config.device)
    )
    data_limited += noise

    fbp_recon = radon_limited.backward(filter_sinogram(data_limited))
    x0 = torch.zeros([config.N, config.N]).to(config.device)

    landweber_limited = torch_radon.solvers.Landweber(
        radon_limited, projection=None, grad=False
    )

    if example == "synthetic":
        lam = 1e-5
    else:
        lam = 1e-6
    landweber_recon = landweber_limited.run(x0, data_limited, lam, iterations=1000)

    if example == "synthetic":
        L = 500
        tv_recon = tv(
            x0=x0,
            A=radon_limited,
            g=data_limited,
            L=L,
            Niter=1000,
            alpha=0.05,
            tau=1 / L,
            sigma=1 / L,
            theta=1,
            ground_truth=image,
        )

        p_0 = 0.0005
        p_1 = 0.05
        Niter = 100
        shearlet = ShearletTransform(config.N, config.N, [0.5] * 3)
        ell1_recon = ell1_shearlet(
            shearlet=shearlet,
            A=radon_limited,
            sinogram=data_limited,
            p_0=p_0,
            p_1=p_1,
            Niter=Niter,
            ground_truth=None,
            print_flag=False,
        )
    else:
        L = 500
        tv_recon = tv(
            x0=x0,
            A=radon_limited,
            g=data_limited,
            L=L,
            Niter=2000,
            alpha=0.5,
            tau=1 / L,
            sigma=1 / L,
            theta=1,
            ground_truth=image,
        )

        from src.algorithms.ell1_wavelet import ell1_wavelet

        wavelet = DWTForward(J=5, mode="zero", wave="db3").cuda()
        iwavelet = DWTInverse(mode="zero", wave="db3").cuda()

        ell1_recon = ell1_wavelet(
            wavelet,
            iwavelet,
            radon_limited,
            data_limited,
            p_0=0.00001,
            p_1=0.001,
            Niter=100,
            ground_truth=None,
            print_flag=False,
        )

    recon_err = mse(image.cpu().numpy(), tv_recon.cpu().numpy())
    print(f"TV RECON ERROR {recon_err}")
    # print(f"BETA: {beta}")
    return (image, data_limited, fbp_recon, tv_recon, ell1_recon, landweber_recon)


def linear_regression(y0, z0, lr=0.0001, num_iterations=100):

    torch.manual_seed(0)
    # Initialize a and b as trainable parameters
    a = torch.rand(1, requires_grad=True, device="cuda")
    b = torch.rand(1, requires_grad=True, device="cuda")
    # b = b * 0

    # Define optimizer
    optimizer = torch.optim.SGD([a, b], lr=lr)
    # optimizer = torch.optim.SGD([a], lr=lr)

    # Training loop to minimize L2 norm
    for _ in range(num_iterations):
        optimizer.zero_grad()

        # Model prediction: y_pred = a * z0 + b
        y_pred = a * z0 + b
        # y_pred = a * z0

        # Loss function: L2 norm (mean squared error)
        loss = torch.nn.functional.mse_loss(y_pred, y0)
        # print(loss.item())

        # Backpropagate the gradient and optimize
        loss.backward()
        optimizer.step()

    return a.item(), b.item()


def add_circle(image, center, radius, value=1):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (i - center[0]) ** 2 + (j - center[1]) ** 2 < radius**2:
                image[i, j] = value
    return image


def add_square(image, top_left, size, value=1):
    image[top_left[0] : top_left[0] + size, top_left[1] : top_left[1] + size] = value
    return image


def add_ellipse(image, center, axes, value=1):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (
                ((i - center[0]) / axes[0]) ** 2 + ((j - center[1]) / axes[1]) ** 2
            ) <= 1:
                image[i, j] = value
    return image


def add_gaussian(image, center, sigma):
    x = np.arange(0, image.shape[0], 1)
    y = np.arange(0, image.shape[1], 1)
    x, y = np.meshgrid(x, y)
    gaussian = np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / (2 * sigma**2))
    image += gaussian
    return image


if __name__ == "__main__":
    # Parameters
    height = 362
    width = 362
    num_ellipses = 12

    # Generate the image
    # image = generate_random_ellipses_image_grayscale(
    #     height, width, num_ellipses
    # )
    # Image size
    N = height
    image = np.zeros((N, N))

    # Add shapes evenly spread across the array
    image = add_circle(
        image, center=(N // 4, N // 4), radius=30, value=1
    )  # Circle in top-left
    image = add_square(
        image, top_left=(2 * N // 3, 2 * N // 3), size=50, value=0.8
    )  # Square in bottom-right
    image = add_ellipse(
        image, center=(2 * N // 3, N // 4), axes=(30, 50), value=0.6
    )  # Ellipse in top-right

    # Add a Gaussian bell curve in bottom-left
    image = add_gaussian(image, center=(2 * N // 3, N // 4), sigma=15)

    # Apply Gaussian smoothing to the whole image for added effect
    image_smoothed = gaussian_filter(image, sigma=2)

    # Display the result
    plt.imshow(image_smoothed, cmap="gray")
    plt.colorbar()
    plt.title("Evenly Spread Geometric Shapes with Gaussian Bell Curve")
    plt.show()

    # Display the image
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.savefig("ellipses.png")
