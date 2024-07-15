import torch
from abc import ABC, abstractmethod


class PDEFunc(ABC):
    def __init__(self, device='cpu'):
        """ Given u and derivatives, return the PDE residual"""
        pass

    @abstractmethod
    def residuals(self, u_dus: tuple[torch.Tensor, ...], Xs: torch.Tensor, thetas=None) -> torch.Tensor:
        pass


class Poisson(PDEFunc):
    def __init__(self, device='cpu'):
        super().__init__(device=device)

    def residuals(self, u_dus: tuple[torch.Tensor, ...], Xs: torch.Tensor, thetas=None):
        """
        Args:
            u_dus: Values of u, du/dx, d2u/dx2
            Xs: Grid points
            thetas: model parameters

        f(u, du/dX, d2u/dX2, X, thetas) = 0

        dudX.shape = [N_dim, N, N]
        Returns: PDE residual (=0 for exact solution), shape=[N, N]

        """

        u, dudX, d2udX2 = u_dus

        # print(f'{u.shape = }, {dudX.shape = }, {d2udX2.shape = }')

        return d2udX2[1] + 5 * u


