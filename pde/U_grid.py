import torch
from torch import Tensor
import abc
from X_grid import XGrid


class UGrid(abc.ABC):
    N_dim: int

    us: Tensor
    Xs: Tensor

    dirichlet_bc: Tensor
    grad_mask: Tensor  # Which us have gradient. Shape = [N+2, ...]
    pde_mask: Tensor  # Which PDEs are used to fit us. Shape = [N, ...]
    u_mask: tuple[slice, ...]  # Real us points
    N_us_train: int | Tensor

    def __init__(self, X_grid: XGrid):
        self.device = X_grid.device
        self.Xs = X_grid.Xs
        self.dx = X_grid.dx
        self.N = X_grid.N

    def get_us_trainable(self):
        """
        Given an array, remove points that don't have gradient.
        """
        us_grad = self.us[self.grad_mask]
        return us_grad

    def add_nograd_to_us(self, us_grad):
        """
        Add points that don't have gradient to us. Used for jacobian computation.
        """
        us_all = torch.clone(self.us)
        us_all[self.grad_mask] = us_grad
        return us_all

    def update_grid(self, deltas):
        """
        deltas.shape = [N]
        us -> us - deltas
        """
        self.us[self.grad_mask] -= deltas

    def set_grid(self, new_us):
        self.us[self.grad_mask] = new_us

    def get_real_us_Xs(self):
        """ Return all actual grid points, excluding fake boundaries. """
        return self.us[self.u_mask], self.Xs[self.u_mask]

    def get_all_us_Xs(self):
        """ Return all grid points, including fake boundaries. """
        return self.us, self.Xs

    def _cuda(self):
        self.us = self.us.cuda()
        self.Xs = self.Xs.cuda()
        self.dirichlet_bc = self.dirichlet_bc.cuda()
        self.grad_mask = self.grad_mask.cuda()
        self.pde_mask = self.pde_mask.cuda()


class UGrid1D(UGrid):
    """
    Contains grid of points for solving PDE.
    Handles and saves boundary conditions and masks.
    """
    pre_approved_bc: set = {'x0_lower', 'x0_upper'}
    neuman_bc: dict
    dirichlet_bc: dict

    def __init__(self, X_grid: XGrid, dirichlet_bc: dict = None, neuman_bc: dict = None, device='cpu'):
        super().__init__(X_grid, device)
        self.N_dim = 1
        self._init_bc_dict(dirichlet_bc, neuman_bc)

        self.dx = X_grid.dx

    def _init_bc_dict(self, dirichlet_bc, neuman_bc):
        if dirichlet_bc is None:
            dirichlet_bc = dict()
        if neuman_bc is None:
            neuman_bc = dict()
        assert set(dirichlet_bc.keys()).issubset(self.pre_approved_bc), f'{dirichlet_bc.keys()} not in {self.pre_approved_bc}'
        assert set(neuman_bc.keys()).issubset(self.pre_approved_bc), f'{neuman_bc.keys()} not in {self.pre_approved_bc}'
        self.dirichlet_bc = dirichlet_bc
        self.neuman_bc = neuman_bc

    def remove_bc(self, us):
        """
        Given an array, remove points that don't have gradient.
        """
        us = us[self.u_mask]
        return us

    def update_grid(self, deltas):
        """
        deltas.shape = [N]
        us -> us - deltas
        """
        self.us[self.grad_mask] -= deltas

    def set_grid(self, new_us):
        self.us[self.grad_mask] = new_us

    def get_real_u_x(self):
        """ Return all actual grid points, excluding fake boundaries. """
        return self.us[self.u_mask], self.xs[self.u_mask]

    @abc.abstractmethod
    def get_with_bc(self):
        """
        Return grid of points with boundary conditions set.
        """
        pass

    def __repr__(self):
        return f"Grid of values, {self.us}"


class UGridOpen1D(UGrid1D):
    def __init__(self, X_grid: XGrid,
                 dirichlet_bc: dict = None, neuman_bc: dict = None,
                 device='cpu'):
        """
        :param L: Size of domain
        :param N: Number of points PDE is enforced at
        :param dirichlet_bc: Dict of {'left'/'right': Value at bc}. Set inside the grid
        :param neuman_bc: Dict of {'left'/'right': Value at bc}. Set outside the grid

        Add on extra terms at end of grid so derivatives can be taken at the boundary if needed.
        PDE is constrained at N middle points (u0 - un-1), but derivatives are taken at N - N_boundary points.
        Format of grid:
            Only dirichlet:                 [E, u0=DL, u1,..., un-1=DR, E]
            Dirichlet and Neuman on left:   [NL, u0=DL, u1,..., un-1, E]
            Dirichlet left, Neuman right:   [E, u0=DL, u1,..., un-1, NR]
            Neuman only:                    [NL, u0, u1,..., un-1, NR]
        """
        super().__init__(X_grid, dirichlet_bc, neuman_bc, device)

        # Grid of points.
        self.Xs = X_grid.xs
        self.us = torch.zeros_like(self.Xs).to(self.device)  # Shape: [N + 2]

        # Mask of where to compute gradients, no gradient at boundary
        self.grad_mask = torch.ones_like(self.Xs).to(torch.bool).to(self.device)
        if 'x0_lower' in self.dirichlet_bc:
            self.grad_mask[1] = 0
        if 'x0_upper' in self.dirichlet_bc:
            self.grad_mask[-2] = 0
        if 'x0_lower' in self.neuman_bc:
            self.grad_mask[0] = 0
        if 'x0_upper' in self.neuman_bc:
            self.grad_mask[-1] = 0

        # Mask of which elements are real
        self.u_mask = (slice(1, -1),)

    def get_with_bc(self):
        if 'x0_lower' in self.dirichlet_bc:
            self.us[1] = self.dirichlet_bc['x0_lower']
        if 'x0_upper' in self.dirichlet_bc:
            self.us[-2] = self.dirichlet_bc['x0_upper']

        # Compute imaginary value so derivatives are as given.
        if 'x0_lower' in self.neuman_bc:
            self.us[0] = self.us[2] - 2 * self.neuman_bc['x0_lower'] * self.dx
        if 'x0_upper' in self.neuman_bc:
            self.us[-1] = self.us[-3] + 2 * self.neuman_bc['x0_upper'] * self.dx
        return self.us


class UGridClosed1D(UGrid1D):
    """
    Like a PDEGrid, but setting neuman boundary conditions with interior points instead of fantasy.
    """

    def __init__(self, X_grid: XGrid, dirichlet_bc: dict = None, neuman_bc: dict = None, device='cpu'):
        """
        :param L: Size of domain
        :param N: Number of points PDE is enforced at
        :param dirichlet_bc: Dict of {'left'/'right': Value at bc}. Set inside the grid
        :param neuman_bc: Dict of {'left'/'right': Value at bc}. Set outside the grid
        """
        super().__init__(X_grid, dirichlet_bc, neuman_bc, device)

        # Grid of points.
        self.xs = X_grid.xs[1:-1]
        self.us = torch.ones_like(self.xs).to(self.device)  # Shape: [N]

        # Mask of where to compute gradients, no gradient at boundary
        self.grad_mask = torch.ones_like(self.xs).to(torch.bool).to(self.device)
        if 'x0_lower' in self.dirichlet_bc:
            self.grad_mask[0] = 0
        if 'x0_upper' in self.dirichlet_bc:
            self.grad_mask[-1] = 0
        if 'x0_lower' in self.neuman_bc:  # Don't use imaginary boundary if not needed
            self.grad_mask[1] = 0
        if 'x0_upper' in self.neuman_bc:
            self.grad_mask[-2] = 0

        # Mask of which elements are real
        self.u_mask = torch.ones_like(self.xs).to(torch.bool).to(self.device)

    def get_with_bc(self):
        if 'x0_lower' in self.dirichlet_bc:
            self.us[0] = self.dirichlet_bc['x0_lower']
        if 'x0_upper' in self.dirichlet_bc:
            self.us[-1] = self.dirichlet_bc['x0_upper']

        # Use second order gradient approximation on boundary.
        if 'x0_lower' in self.neuman_bc:
            self.us[1] = 1 / 4 * (2 * self.neuman_bc['x0_lower'] * self.dx + 3 * self.us[0] + self.us[2])
        if 'x0_upper' in self.neuman_bc:
            self.us[-2] = 1 / 4 * (2 * self.neuman_bc['x0_upper'] * self.dx - 3 * self.us[-1] - self.us[-3])

        return self.us


class UGrid2D(UGrid):
    """
    Contains grid of points for solving PDE.
    Handles and saves boundary conditions and masks.
    """

    def __init__(self, X_grid: XGrid, dirichlet_bc: Tensor = None, neuman_bc: Tensor = None):
        """
        Boundary conditions set as a tensor of shape [Nx, Ny], with NaN for no boundary, including center.
        """
        super().__init__(X_grid)
        self.N_dim = 2

        self._init_bc_dict(dirichlet_bc, neuman_bc)

    def _init_bc_dict(self, dirichlet_bc, neuman_bc):
        if dirichlet_bc is None:
            dirichlet_bc = torch.full(self.N.tolist(), float("nan"))
        else:
            assert torch.all(torch.isnan(dirichlet_bc[1:-1, 1:-1])), f'{dirichlet_bc = } is not NaN in middle'

        if neuman_bc is None:
            neuman_bc = torch.full(self.N.tolist(), float("nan"))
        else:
            assert torch.all(torch.isnan(neuman_bc[1:-1, 1:-1])), f'{neuman_bc = } is not NaN in middle'

        self.dirichlet_bc = dirichlet_bc
        self.neuman_bc = neuman_bc

    @abc.abstractmethod
    def _fix_bc(self):
        pass

    def __repr__(self):
        return f"Grid of values, {self.us}"


class UGridOpen2D(UGrid2D):
    def __init__(self, X_grid: XGrid, dirichlet_bc: Tensor = None, neuman_bc: Tensor = None):
        super().__init__(X_grid, dirichlet_bc, neuman_bc)

        # Real elements
        self.u_mask = (slice(1, -1), slice(1, -1))

        # Select which boundary equations are needed to enforce PDE at to save computation
        pde_mask = torch.zeros(*X_grid.N).to(torch.bool)
        self.pde_mask = pde_mask | torch.isnan(self.neuman_bc) | torch.isnan(self.dirichlet_bc)
        # self.pde_mask = torch.zeros(self.Xs.shape[:2]).bool()
        #self.pde_mask[1:-1, 1:-1] = pde_mask
        #self.pde_mask = pde_mask

        self.us = torch.zeros(self.Xs.shape[:2]).to(self.device)  # Shape: [N + 2]
        # Set corner to NaN. May need to be changed if using 9 point stencil
        self.us[0, 0] = float("nan")
        self.us[-1, -1] = float("nan")
        self.us[0, -1] = float("nan")
        self.us[-1, 0] = float("nan")

        # Mask of Dirichlet boundary conditions. Extended to include boundary points
        dirichlet_bc = torch.full_like(self.us, float("nan"))
        dirichlet_bc[1:-1, 1:-1] = self.dirichlet_bc
        self.dirichlet_bc = dirichlet_bc

        self.dirichlet_mask = ~torch.isnan(self.dirichlet_bc)

        # Mask of Neuman boundary conditions. Extended to include boundary points
        neuman_bc = torch.full_like(self.us, float("nan"))
        neuman_bc[1:-1, 1:-1] = self.neuman_bc
        self.neuman_bc = neuman_bc

        self.neuman_mask = torch.zeros_like(self.us).to(torch.bool)
        # Left
        self.neuman_mask[0] = ~torch.isnan(self.neuman_bc[1, :])
        # Right
        self.neuman_mask[-1] = ~torch.isnan(self.neuman_bc[-2, :])
        # Top
        self.neuman_mask[:, -1] = ~torch.isnan(self.neuman_bc[:, -2])
        # Bottom
        self.neuman_mask[:, 0] = ~torch.isnan(self.neuman_bc[:, 1])

        # Fix corners with Neuman (as Nans)
        self.neuman_mask[[0, 0, -1, -1], [0, -1, 0, -1]] = True

        # Mask of unfixed elements that require grad
        self.grad_mask = ~(self.neuman_mask | self.dirichlet_mask)
        self.N_us_train = self.grad_mask.sum()

        # TODO: Dirichlet corner conditions
        self.pde_mask[0, 0] = False
        self.grad_mask[1, 0] = False
        self.grad_mask[0, 1] = False

        self.pde_mask[-1, -1] = False
        self.grad_mask[-2, -1] = False
        self.grad_mask[-1, -2] = False

        self.pde_mask[0, -1] = False
        self.grad_mask[1, -1] = False
        self.grad_mask[0, -2] = False

        self.pde_mask[-1, 0] = False
        self.grad_mask[-2, 0] = False
        self.grad_mask[-1, 1] = False

        N_us_train, N_us = self.grad_mask.sum().item(), self.pde_mask.sum().item()
        assert N_us_train == N_us, f"Need as many equations as unknowns, {N_us_train = } != {N_us = }"
        self._fix_bc()

        if self.device == 'cuda':
            self._cuda()

    def _fix_bc(self):
        # Dirichlet BC
        self.us[self.dirichlet_mask] = self.dirichlet_bc[self.dirichlet_mask]

        # Neuman BC. Setting imaginary values so derivatives on boundary are as given.
        # Left
        self.us[0, self.neuman_mask[0]] = self.us[2, self.neuman_mask[0]] - 2 * self.neuman_bc[1, self.neuman_mask[0]] * self.dx
        # Right
        self.us[-1, self.neuman_mask[-1]] = self.us[-3, self.neuman_mask[-1]] + 2 * self.neuman_bc[-2, self.neuman_mask[-1]] * self.dx
        # Top
        self.us[self.neuman_mask[:, -1], -1] = self.us[self.neuman_mask[:, -1], -3] + 2 * self.neuman_bc[self.neuman_mask[:, -1], -2] * self.dx
        # Bottom
        self.us[self.neuman_mask[:, 0], 0] = self.us[self.neuman_mask[:, 0], 2] - 2 * self.neuman_bc[self.neuman_mask[:, 0], 1] * self.dx


if __name__ == "__main__":
    x = UGrid2D
