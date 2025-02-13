# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_radon
from alpha_transform import AlphaShearletTransform
from alpha_transform import AlphaShearletTransform as AST
from alpha_transform.fourier_util import my_ifft_shift


def _normalize_shape(x, d):
    old_shape = x.size()[:-d]
    x = x.view(-1, *(x.size()[-d:]))
    return x, old_shape


def _unnormalize_shape(y, old_shape):
    if isinstance(y, torch.Tensor):
        y = y.view(*old_shape, *(y.size()[1:]))
    elif isinstance(y, tuple):
        y = [yy.view(*old_shape, *(yy.size()[1:])) for yy in y]

    return y


def normalize_shape(d):
    """
    Input with shape (batch_1, ..., batch_n, s_1, ..., s_d) is reshaped to (batch, s_1, s_2, ...., s_d)
    fed to f and output is reshaped to (batch_1, ..., batch_n, s_1, ..., s_o).
    :param d: Number of non-batch dimensions
    """

    def wrap(f):
        def wrapped(self, x, *args, **kwargs):
            x, old_shape = _normalize_shape(x, d)

            y = f(self, x, *args, **kwargs)

            return _unnormalize_shape(y, old_shape)

        wrapped.__doc__ = f.__doc__
        return wrapped

    return wrap


class ShearletTransform:
    """
    Implementation of Alpha-Shearlet transform based on https://github.com/dedale-fet/alpha-transform/tree/master/alpha_transform.

    Once the shearlet spectrograms are computed all the computations are done on the GPU.

    :param width: Width of the images
    :param height: Height of the images
    :param alphas: List of alpha coefficients that will be used to generate shearlets
    :param cache: If specified it should be a path to a directory that will be used to cache shearlet coefficients in
        order to avoid recomputing them at each instantiation of this class.

    .. note::
        Support both float and double precision.
    """

    def __init__(self, width, height, alphas, cache=None):
        cache_name = f"{width}_{height}_{alphas}.npy"
        if cache is not None:
            if not os.path.exists(cache):
                os.makedirs(cache)

            cache_file = os.path.join(cache, cache_name)
            if os.path.exists(cache_file):
                shifted_spectrograms = np.load(cache_file)
            else:
                alpha_shearlet = AlphaShearletTransform(width, height, alphas, real=True, parseval=True)
                shifted_spectrograms = np.asarray([my_ifft_shift(spec) for spec in alpha_shearlet.spectrograms])
                np.save(cache_file, shifted_spectrograms)
        else:
            alpha_shearlet = AlphaShearletTransform(width, height, alphas, real=True, parseval=True)
            scales = [0] + [x[0] for x in alpha_shearlet.indices[1:]]
            self.scales = np.asarray(scales)
            shifted_spectrograms = np.asarray([my_ifft_shift(spec) for spec in alpha_shearlet.spectrograms])

        self.scales = torch.FloatTensor(self.scales)
        self.shifted_spectrograms = torch.FloatTensor(shifted_spectrograms)

        self.shifted_spectrograms_d = torch.DoubleTensor(shifted_spectrograms)

    def _move_parameters_to_device(self, device):
        if device != self.shifted_spectrograms.device:
            self.shifted_spectrograms = self.shifted_spectrograms.to(device)
            self.shifted_spectrograms_d = self.shifted_spectrograms_d.to(device)

    # @normalize_shape(2)
    def forward(self, x):
        """
        Do shearlet transform of a batch of images.

        :param x: PyTorch GPU tensor with shape :math:`(d_1, \\dots, d_n, h, w)`.
        :returns: PyTorch GPU tensor containing shearlet coefficients.
            Has shape :math:`(d_1, \\dots, d_n, \\text{n_shearlets}, h, w)`.
        """
        self._move_parameters_to_device(x.device)

        # c = torch.fft.rfft(x, 2, norm="ortho")
        c = torch.fft.fft2(x, norm="ortho")  # , 2, norm="ortho")#.unsqueeze(0)
        cs = torch.einsum("fij,ij->fij", self.shifted_spectrograms, c)

        # if x.dtype == torch.float64:
        # cs = torch.einsum("fij,bijc->bfijc", self.shifted_spectrograms_d, c)
        # else:
        # cs = torch.einsum("fij,bijc->bfijc", self.shifted_spectrograms, c)
        return torch.fft.ifft2(cs, norm="ortho")
        # return torch.fft.irfft2(cs, norm="ortho")

    # @normalize_shape(3)
    def backward(self, cs):
        """
        Do inverse shearlet transform.

        :param cs: PyTorch GPU tensor containing shearlet coefficients,
            with shape :math:`(d_1, \\dots, d_n, \\text{n_shearlets}, h, w)`.
        :returns: PyTorch GPU tensor containing reconstructed images.
            Has shape :math:`(d_1, \\dots, d_n, h, w)`.
        """

        # cs_fft = torch.fft.rfft(cs, 2, norm="ortho").unsqueeze(0)
        cs_fft = torch.fft.fft2(cs, norm="ortho")
        res = torch.einsum("fij,fij->ij", self.shifted_spectrograms.type(torch.complex64), cs_fft)
        # if cs.dtype == torch.float64:
        #     res = torch.einsum("fij,bfijc->bijc", self.shifted_spectrograms_d, cs_fft)
        # else:
        #     res = torch.einsum("fij,bfijc->bijc", self.shifted_spectrograms.type(torch.complex64), cs_fft)
        # return torch.fft.irfft(res, 2, norm="ortho")
        return torch.real(torch.fft.ifft2(res, norm="ortho"))


if __name__ == "__main__":
    width = 512
    height = 512
    alphas = [0.5] * 2
    ST = ShearletTransform(width, height, [0.5] * 4, cache=None)

    f = np.load("data_htc2022_simulated/images_without_blur.npy")[1].astype("float64")
    f /= np.max(f)
    f += 0.08 * np.random.randn(*f.shape)

    alpha = 0.1
    c = ST.forward(torch.Tensor(f))
    c = torch.sgn(c) * torch.maximum(torch.tensor(0.01), torch.abs(c) - alpha)
    rec = ST.backward(c)

    plt.figure()
    plt.imshow(torch.real(rec).cpu())
    plt.show()
    plt.colorbar()
    plt.savefig("shearlet_test1.png")
