import torch
import torch.nn as nn


class range_layer(nn.Module):
    """
    A PyTorch layer that projects the input tensor onto dom(A).

    Methods:
    --------
    forward(x: torch.Tensor, A: nn.Module, B: nn.Module) -> torch.Tensor
        Applies operations A and B sequentially to each sample in the batch
        and returns the result.

    Parameters:
    -----------
    x : torch.Tensor
        Input tensor of shape (batch_size, channels, height, width).

    A : nn.Module
        The forward operator.

    B : nn.Module
        The Moore-Penrose Inverse implementation of A.

    Returns:
    --------
    torch.Tensor
        The projected tensor onto dom(A) calculate by B(A(x)).
    """

    def __init__(self):
        super(range_layer, self).__init__()

    def forward(
        self, x: torch.Tensor, A: nn.Module, B: nn.Module
    ) -> torch.Tensor:
        Z = torch.zeros_like(x)
        for j in range(x.shape[0]):
            Z[j, 0, :, :] = B(A(x[j, 0, :, :]))
        return Z


class null_space_layer(nn.Module):
    """
    A PyTorch layer that projects `x` onto the null space of the forward
    operator.

    Methods:
    --------
    forward(x: torch.Tensor, A: nn.Module, B: nn.Module) -> torch.Tensor
        Computes the residual for each sample in the batch.

    Parameters:
    -----------
    x : torch.Tensor
        Input tensor of shape (batch_size, channels, height, width).

    A : nn.Module
        The forward operator.

    B : nn.Module
        The Moore-Penrose Inverse implementation of A.

    Returns:
    --------
    torch.Tensor
        The projection onto the null space by calculating x - B(A(x)).
    """

    def __init__(self):
        super(null_space_layer, self).__init__()

    def forward(
        self, x: torch.Tensor, A: nn.Module, B: nn.Module
    ) -> torch.Tensor:
        Z = torch.zeros_like(x)
        for j in range(x.shape[0]):
            Z[j, 0, :, :] = x[j, 0, :, :] - B(A(x[j, 0, :, :]))
        return Z


class proximal_layer(nn.Module):
    """
    A PyTorch layer that applies the data-proximal projection onto the
    orthogonal complement of ker(A).

    Attributes:
    -----------
    ell2_norm : float
        The norm threshold used in the proximal operator.

    Methods:
    --------
    Phi(x: torch.Tensor) -> torch.Tensor
        Applies the proximal function to the input tensor.

    forward(x: torch.Tensor, A: nn.Module, B: nn.Module) -> torch.Tensor
        Applies A, Phi and B to each sample in the batch.

    Parameters:
    -----------
    x : torch.Tensor
        Input tensor of shape (batch_size, channels, height, width).

    A : nn.Module
        The forward operator.

    B : nn.Module
        The Moore-Penrose Inverse implementation of A.

    Returns:
    --------
    torch.Tensor
        The data-proximal projection calculated by B(Phi(A(x))).
    """

    def __init__(self, ell2_norm: float):
        super(proximal_layer, self).__init__()
        self.ell2_norm = ell2_norm

    def Phi(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.zeros_like(x)
        norm = torch.linalg.norm(x)
        if norm < self.ell2_norm:
            y = x
        else:
            y = self.ell2_norm * x / torch.sqrt(norm**2)
        return y

    def forward(
        self, x: torch.Tensor, A: nn.Module, B: nn.Module
    ) -> torch.Tensor:
        Z = torch.zeros_like(x)
        for j in range(x.shape[0]):
            y = A(x[j, 0, :, :])
            y = self.Phi(y)
            z = B(y)
            Z[j, 0, :, :] = z
        return Z
