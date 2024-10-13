import torch
from torch import Tensor
import abc
from pde.cartesian_grid.X_grid import XGrid
from pde.utils import adjust_slice, show_grid

class UBase(abc.ABC):
    device: torch.device | str
    N_dim: int
    N_points: int  # Number of real points.
    dirichlet_bc: Tensor
    neuman_bc: Tensor

    grad_mask: Tensor  # Which us have gradient. Shape = [N+2, ...]
    pde_mask: Tensor  # Which PDEs are used to fit us. Automatically disregard extra points. Shape = [N, ...]
    u_mask: tuple[slice, ...]  # Real us points [1:-1, ...]
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
        self._fix_bc()

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


    def _cuda(self):
        self.us = self.us.cuda(non_blocking=True)
        self.Xs = self.Xs.cuda(non_blocking=True)
        self.dirichlet_bc = self.dirichlet_bc.cuda(non_blocking=True)
        self.grad_mask = self.grad_mask.cuda(non_blocking=True)
        self.pde_mask = self.pde_mask.cuda(non_blocking=True)