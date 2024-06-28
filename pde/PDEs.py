import torch
from abc import ABC, abstractmethod


class PDEFunc(ABC):
    def __init__(self, device='cpu'):
        pass

    @abstractmethod
    def forward(self, Us: torch.Tensor, Xs: torch.Tensor, thetas=None):
        pass


class Poission(PDEFunc):
    def __init__(self, device='cpu'):
        super().__init__(device=device)

    def forward(self, u_dus: tuple[torch.Tensor], Xs: torch.Tensor, thetas=None):
        """
        Args:
            u_dus: Values of u, du/dx, d2u/dx2
            Xs: Grid points
            thetas: model parameters

        f(u, du/dX, d2u/dX2, X, thetas) = 0

        Returns: PDE residual (=0 for exact solution)

        """

        u, dudX, d2udX2 = u_dus

        return d2udX2


