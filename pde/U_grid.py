import torch
from torch import Tensor
import abc
from X_grid import XGrid
from utils import show_grid


def adjust_slice(slice_obj, start_adjust=0, stop_adjust=0):
    """Adjust the given slice object by modifying its start and stop values."""
    new_start = slice_obj.start + start_adjust
    new_stop = slice_obj.stop + stop_adjust
    return slice(new_start, new_stop)


class UGrid(abc.ABC):
    N_dim: int
    N: Tensor  # Number of real points in each dimension
    us: Tensor
    Xs: Tensor

    dirichlet_bc: Tensor
    neuman_bc: Tensor

    grad_mask: Tensor  # Which us have gradient. Shape = [N+2, ...]
    pde_mask: Tensor  # Which PDEs are used to fit us. Automatically disregard extra points. Shape = [N, ...]
    u_mask: tuple[slice, ...]  # Real us points [1:-1, ...]
    N_us_train: int | Tensor

    def __init__(self, X_grid: XGrid):
        self.device = X_grid.device
        self.Xs = X_grid.Xs
        self.dx = X_grid.dx
        self.N = X_grid.N

    def get_us_mask(self):
        """
        Return us, and mask of which elements are trainable. Used for masking Jacobian equations.
        """
        return self.us, self.grad_mask, self.pde_mask

    def add_nograd_to_us(self, us_grad, mask=None):
        """
        Add points that don't have gradient to us. Used for Jacobian computation.
        """
        if mask is None:
            mask = self.grad_mask

        us_all = torch.clone(self.us)
        us_all[mask] = us_grad

        # Neuman BCs also need to be set using us_grad to make sure gradient is tracked across boundary
        self._fix_neuman_bc(us_all)
        return us_all

    def update_grid(self, deltas):
        """
        Update grid with changes, and fix boundary conditions with new grid.
        deltas.shape = [N]
        us -> us - deltas
        """
        self.us[self.grad_mask] -= deltas
        self._fix_bc()

    def get_real_us_Xs(self):
        """ Return all actual grid points, excluding fake boundaries. """
        return self.us[self.u_mask], self.Xs[self.u_mask]

    def get_all_us_Xs(self):
        """ Return all grid points, including fake boundaries. """
        return self.us, self.Xs

    @abc.abstractmethod
    def _fix_bc(self):
        """ Fix boundary conditions. To be called after init and every update of grid."""
        pass

    @abc.abstractmethod
    def _fix_neuman_bc(self, us):
        """ Neuman boundary conditions are special, since autograd needs to track values across two points. """
        pass

    def _cuda(self):
        self.us = self.us.cuda()
        self.Xs = self.Xs.cuda()
        self.dirichlet_bc = self.dirichlet_bc.cuda()
        self.grad_mask = self.grad_mask.cuda()
        self.pde_mask = self.pde_mask.cuda()


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

            corner_idx = [(0, 0), (0, -1), (-1, 0), (-1, -1)]
            corners = torch.stack([neuman_bc[idx] for idx in corner_idx])
            assert torch.all(torch.isnan(corners)), f'{neuman_bc = } Undefined Neuman corner values.'

        self.dirichlet_bc = dirichlet_bc
        self.neuman_bc = neuman_bc

    def __repr__(self):
        return f"Grid of values, {self.us}"


class UGridOpen2D(UGrid2D):
    def __init__(self, X_grid: XGrid, dirichlet_bc: Tensor = None, neuman_bc: Tensor = None):
        super().__init__(X_grid, dirichlet_bc, neuman_bc)

        # Real elements
        self.u_mask = (slice(1, -1), slice(1, -1))

        n_us = (self.N + 2).tolist()
        self.us = torch.zeros(n_us, dtype=torch.float32).to(self.device)

        # Mask of Dirichlet boundary conditions. Extended to include boundary points
        dirichlet_bc = torch.full_like(self.us, float("nan"))
        dirichlet_bc[1:-1, 1:-1] = self.dirichlet_bc
        self.dirichlet_bc = dirichlet_bc
        self.dirichlet_mask = ~torch.isnan(dirichlet_bc)

        # Mask of Neuman boundary conditions. Extended to include boundary points
        neuman_bc = torch.full_like(self.us, float("nan"))
        neuman_bc[1:-1, 1:-1] = self.neuman_bc
        self.neuman_bc = neuman_bc
        self.neuman_mask = torch.zeros_like(self.us).to(torch.bool)
        # Left
        self.neuman_mask[0] = ~torch.isnan(neuman_bc[1, :])
        # Right
        self.neuman_mask[-1] = ~torch.isnan(neuman_bc[-2, :])
        # Top
        self.neuman_mask[:, -1] = ~torch.isnan(neuman_bc[:, -2])
        # Bottom
        self.neuman_mask[:, 0] = ~torch.isnan(neuman_bc[:, 1])

        # Select which boundary equations are needed to enforce PDE at to save computation
        self.grad_mask = torch.zeros_like(self.us).to(torch.bool)
        self.grad_mask[2:-2, 2:-2] = True

        self.pde_mask = torch.zeros_like(self.us).to(torch.bool)
        self.pde_mask[2:-2, 2:-2] = True
        edge_idxs = [((0, slice(1, -1)), (1, slice(1, -1))),  # Left
                     ((-1, slice(1, -1)), (-2, slice(1, -1))),  # Right
                     ((slice(1, -1), -1), (slice(1, -1), -2)),  # Top
                     ((slice(1, -1), 0), (slice(1, -1), 1))  # Bottom
                     ]

        for border_idx, inset_idx in edge_idxs:
            pde_vals, grad_vals = self._set_bc_masks_1D(self.neuman_mask[border_idx], self.dirichlet_mask[inset_idx])

            self.pde_mask[inset_idx] = pde_vals
            self.grad_mask[inset_idx] = grad_vals

        self.N_us_train = self.grad_mask.sum()
        N_us_train, N_us = self.N_us_train.item(), self.pde_mask.sum().item()
        assert N_us_train == N_us, f"Need as many equations as unknowns, {N_us_train = } != {N_us = }"

        self._fix_bc()

        if self.device == 'cuda':
            self._cuda()

    def _fix_bc(self):
        # Dirichlet BC
        self.us[self.dirichlet_mask] = self.dirichlet_bc[self.dirichlet_mask]
        self._fix_neuman_bc(self.us)

    def _fix_neuman_bc(self, us):
        # Neuman BC. Setting imaginary values so derivatives on boundary are as given.

        # Left
        us[0, self.neuman_mask[0]] = us[2, self.neuman_mask[0]] + 2 * self.neuman_bc[1, self.neuman_mask[0]] * self.dx
        # Right
        us[-1, self.neuman_mask[-1]] = us[-3, self.neuman_mask[-1]] + 2 * self.neuman_bc[-2, self.neuman_mask[-1]] * self.dx
        # Top
        us[self.neuman_mask[:, -1], -1] = us[self.neuman_mask[:, -1], -3] + 2 * self.neuman_bc[self.neuman_mask[:, -1], -2] * self.dx
        # Bottom
        us[self.neuman_mask[:, 0], 0] = us[self.neuman_mask[:, 0], 2] + 2 * self.neuman_bc[self.neuman_mask[:, 0], 1] * self.dx

    def _set_bc_masks_1D(self, neuman_mask, dirichlet_mask):
        """ Set 1D masks, call this for each boundary.
            Input: Sections of boundary conditions.
            Returns: PDE mask and gradient mask (for 1 step inwards). """

        assert neuman_mask.shape == dirichlet_mask.shape

        pde_mask = neuman_mask & ~dirichlet_mask
        grad_mask = (neuman_mask & ~dirichlet_mask) | (~neuman_mask & ~dirichlet_mask)

        return pde_mask, grad_mask


class USubGrid:
    neuman_mask: torch.Tensor
    neuman_bc: torch.Tensor

    us_grad_mask: torch.Tensor
    pde_mask: torch.Tensor

    us_region: torch.Tensor
    us_pde: torch.Tensor
    Xs_pde: torch.Tensor

    def __init__(self, device, dx):
        self.device = device
        self.dx = dx

    def _fix_neuman_bc(self, us):
        # Neuman BC. Setting imaginary values so derivatives on boundary are as given.
        # Left
        us[0, self.neuman_mask[0]] = us[2, self.neuman_mask[0]] + 2 * self.neuman_bc[1, self.neuman_mask[0]] * self.dx
        # Right
        us[-1, self.neuman_mask[-1]] = us[-3, self.neuman_mask[-1]] + 2 * self.neuman_bc[-2, self.neuman_mask[-1]] * self.dx
        # Top
        us[self.neuman_mask[:, -1], -1] = us[self.neuman_mask[:, -1], -3] + 2 * self.neuman_bc[self.neuman_mask[:, -1], -2] * self.dx
        # Bottom
        us[self.neuman_mask[:, 0], 0] = us[self.neuman_mask[:, 0], 2] + 2 * self.neuman_bc[self.neuman_mask[:, 0], 1] * self.dx

    def get_us_grad(self):
        """
        Return us that have gradient. In flattened format.
        """
        return self.us_region[self.us_grad_mask]

    def add_nograd_to_us(self, us_grad):
        """
        Add points that don't have gradient to us. Used for Jacobian computation.
        """

        us_all = torch.clone(self.us_region)
        us_all[self.us_grad_mask] = us_grad

        # Neuman BCs also need to be set using us_grad to make sure gradient is tracked across boundary
        self._fix_neuman_bc(us_all)
        return us_all


class USplitGrid(USubGrid):
    """  Transformed grid of Us to calculate derivatives.
        Handles masking regions of grid to solve PDE.
    """

    def __init__(self, us_grid: UGrid, region_mask: tuple[slice, ...], us_grad_mask: torch.Tensor, pde_mask: torch.Tensor):
        """ region_mask: Region of grid to calculate PDE in.
            us_grad_mask: Mask of which us have gradient, over full grid. Shape = [N+2, ...]
            pde_mask: Mask of which PDEs are used to fit us, over full grid. Shape = [N+2, ...]

            Note both us_grad_mask and pde_mask have been set to False outside region_mask, but are full arrays.
            """
        super().__init__(us_grid.device, us_grid.dx)

        self.region_mask = region_mask
        # Region PDE is calculated is 1 unit inward from region_mask
        self.inward_mask = tuple([adjust_slice(s, start_adjust=1, stop_adjust=-1) for s in region_mask])

        # Us in region and us used for PDE and Xs for PDE
        all_us, all_Xs = us_grid.get_all_us_Xs()
        self.us_region = all_us[region_mask]
        self.us_pde = all_us[self.inward_mask]
        self.Xs_pde = all_Xs[self.inward_mask].permute(2, 0, 1)

        # Boundary conditions of region
        # Dirichlet is already set, neuman needs to be dynamically set
        self.neuman_mask = us_grid.neuman_mask[region_mask]
        self.neuman_bc = us_grid.neuman_bc[region_mask]

        # Masks for derivatives
        self.us_grad_mask = us_grad_mask[region_mask]  # Mask for which us have gradient and in region
        self.pde_mask = pde_mask[self.inward_mask]  # Mask for PDES in region and selected for grad


class UNormalGrid(USubGrid):
    """ Normal version of subgrid. Effectively placeholder that does nothing. """

    def __init__(self, us_grid: UGrid, us_grad_mask: torch.Tensor, pde_mask: torch.Tensor):
        super().__init__(us_grid.device, us_grid.dx)
        all_us, all_Xs = us_grid.get_all_us_Xs()

        self.us_grad_mask = us_grad_mask
        self.pde_mask = pde_mask[1:-1, 1:-1]

        self.us_region = all_us
        self.Xs_pde = all_Xs[1:-1, 1:-1].permute(2, 0, 1)

        self.neuman_mask = us_grid.neuman_mask
        self.neuman_bc = us_grid.neuman_bc



if __name__ == "__main__":
    x = UGrid2D
