from abc import abstractmethod

import torch
from torch import Tensor
import abc
from pde.cartesian_grid.X_grid import XGrid
from pde.utils import adjust_slice, show_grid

class UBase(abc.ABC):
    device: torch.device | str
    N_dim: int
    N_points: int  # Number of real points.

    us: Tensor  # Value of u at all points.
    Xs: Tensor  # Coordinates of all points. Shape = [N_total, 2]

    grad_mask: Tensor  # Which us have gradient. Shape = [N_u_grad]
    pde_mask: Tensor  # Which PDEs are used to fit us. Automatically disregard extra points. Shape = [N_pde, ...]
    u_mask: tuple[slice, ...]  # Real us points [N_u_real]
    N_us_grad: int        # Number of points that need fitting

    pde_true_idx: Tensor
    us_grad_idx: Tensor

    def get_us_mask(self):
        """
        Return us, and mask of which elements are trainable. Used for masking Jacobian equations.
        """
        return self.us, self.grad_mask, self.pde_mask

    def mask_nonzero_idx(self,):
        """ Tuples of indices where nonzero elements are in masks. """
        return self.pde_true_idx, self.us_grad_idx

    def update_grid(self, deltas):
        """
        Update grid with changes, and fix boundary conditions with new grid.
        deltas.shape = [N]
        us -> us - deltas
        """
        self.us[self.grad_mask] -= deltas
        # self._fix_bc()        # TODO: Fix boundary conditions for graph

    def get_real_us_Xs(self):
        """ Return all actual grid points, excluding fake boundaries. """
        return self.us[self.u_mask], self.Xs[self.u_mask]

    def get_all_us_Xs(self):
        """ Return all grid points, including fake boundaries. """
        return self.us, self.Xs

    # @abc.abstractmethod
    # def _fix_bc(self):
    #     """ Fix boundary conditions. To be called after init and every update of grid."""
    #     pass
    #
    # @abc.abstractmethod
    # def _fix_neuman_bc(self, us):
    #     """ Neuman boundary conditions are special, since autograd needs to track values across two points. """
    #     pass

    @abc.abstractmethod
    def split(self, Ns):
        """ Split computation into subgrids for more efficient Jacobian computation."""
        pass

    def get_us_grad(self):
        """ Return us with gradients. """
        return self.us[self.grad_mask]

    def add_nograd_to_us(self, us_grad):
        """
        Add points that don't have gradient to us. Used for Jacobian computation.
        """

        us_all = torch.clone(self.us)
        us_all[self.grad_mask] = us_grad

        return us_all


    @abstractmethod
    def _cuda(self):
        pass