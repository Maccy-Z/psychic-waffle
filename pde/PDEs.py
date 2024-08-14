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
            Xs: Grid points.
            thetas: model parameters

        f(u, du/dX, d2u/dX2, X, thetas) = 0

        input.shape = [N_dim, Nx, Ny]
        Returns: PDE residual (=0 for exact solution), shape=[Nx, Ny]

        """

        u, dudX, d2udX2 = u_dus

        x_min, x_max = 0.25, 0.5
        y_min, y_max = 0.5, 0.6

        x, y = Xs

        x_masks = (x > x_min) & (x < x_max)
        y_masks = (y > y_min) & (y < y_max)
        charge = 100 * (x_masks & y_masks)

        return d2udX2[1] + d2udX2[0] + charge
