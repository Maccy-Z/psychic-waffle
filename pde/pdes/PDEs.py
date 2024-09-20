import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from pde.config import Config
from pde.utils import show_grid
from pde.X_grid import XGrid2D

class PDEFunc(torch.nn.Module, ABC):
    def __init__(self, cfg: Config, device='cpu'):
        """ Given u and derivatives, return the PDE residual.
            This inits the grid used by the PDE, and boundary conditions.
        """
        super().__init__()
        self.device = device

        grid_N = cfg.N
        N = torch.tensor(cfg.N)
        Xs_grid = XGrid2D(cfg.xmin, cfg.xmax, N, device=cfg.DEVICE)

        self.Xs_grid = Xs_grid

        # Dirichlet BC
        dirichlet_bc = torch.full(grid_N, float('nan'))
        dirichlet_bc[-1, :] = 0.
        # dirichlet_bc[0, :] = 0
        dirichlet_bc[:, 0] = 0.#torch.arange(0, grid_N[0]) / (grid_N[0]-1)
        dirichlet_bc[:, -1] = 0.# torch.arange(0, grid_N[0]) / (grid_N[0]-1)
        self.dirichlet_bc = dirichlet_bc

        # Neuman BC
        neuman_bc = torch.full(grid_N, float('nan'))
        neuman_bc[0, 1:-1] = 0.
        self.neuman_bc = neuman_bc

    def plot_bc(self):
        show_grid(self.dirichlet_bc, "Dirichlet BC")
        show_grid(self.neuman_bc, "Neuman BC")

    def residuals(self, u_dus: tuple[torch.Tensor, ...], Xs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u_dus: u, du/dx, d2u/dx2. Shape = [3][*BS (~Nx * Ny), Nitem].
            Xs: Grid points. Shape = [*BS (~Nx * Ny), 2]

        f(u, du/dX, d2u/dX2, X, thetas) = 0

        Returns: PDE residual (=0 for exact solution), shape=[*BS]
        """
        return self.forward(u_dus, Xs)


class Poisson(PDEFunc):
    def __init__(self, cfg: Config, device='cpu'):
        super().__init__(cfg=cfg, device=device)
        self.to(device)

    def forward(self, u_dus: tuple[torch.Tensor, ...], Xs: torch.Tensor):
        u, dudX, d2udX2 = u_dus
        u = u[..., 0]
        dudx, dudy = dudX[..., 0], dudX[..., 1]
        d2udx2, d2udy2 = d2udX2[..., 0], d2udX2[..., 1]

        resid = 1 * d2udx2 + 1 * d2udy2 + 0 * dudx + 0 * dudy - 5 * u + 5
        return resid

class LearnedFunc(PDEFunc):
    def __init__(self, cfg, device='cpu'):
        super().__init__(cfg=cfg, device=device)

        self.test_param = torch.nn.Parameter(torch.tensor([1.3, 1., 0., 0.], device=device))
        self.to(device)

    def forward(self, u_dus: tuple[torch.Tensor, ...], Xs: torch.Tensor):
        u, dudX, d2udX2 = u_dus
        u = u[..., 0]

        p1, p2, p3, p4 = self.test_param
        resid = p1 * d2udX2[..., 0]  + p2 * d2udX2[..., 1] + p3 * u + p4

        return resid

class NNFunc(PDEFunc):
    def __init__(self, cfg, device='cuda'):
        super().__init__(cfg=cfg, device=device)

        self.lin1 = nn.Linear(5, 32)
        self.lin2 = nn.Linear(32, 1)

        nn.init.zeros_(self.lin2.bias)

        self.to(device)

    def forward(self, u_dus: tuple[torch.Tensor, ...], Xs: torch.Tensor):
        u, dudX, d2udX2 = u_dus

        in_state = torch.cat([u, dudX, d2udX2], dim=-1)
        f = self.lin1(in_state)
        f = self.lin2(f).squeeze()

        return f
