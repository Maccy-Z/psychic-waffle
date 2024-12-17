from abc import abstractmethod

import torch
from torch import Tensor
import abc


class UBase(abc.ABC):
    device: torch.device | str
    N_dim: int
    N_us_real: int  # Number of real points.

    _us: Tensor  # Value of u at all points.
    _Xs: Tensor  # Coordinates of all points. Shape = [N_total, 2]

    grad_mask: Tensor  # Which us have gradient. Shape = [N_u_grad]
    pde_mask: Tensor  # Which PDEs are used to fit us. Automatically disregard extra points. Shape = [N_pde, ...]
    u_mask: tuple[slice, ...]  # Real us points [N_u_real]

    N_us_grad: int        # Number of points that need fitting
    N_component: int           # Number of vector components

    pde_true_idx: Tensor
    us_grad_idx: Tensor



    def mask_nonzero_idx(self,):
        """ Tuples of indices where nonzero elements are in masks. """
        return self.pde_true_idx, self.us_grad_idx

    def update_grid(self, deltas):
        """
        Update grid with changes, and fix boundary conditions with new grid.
        deltas.shape = [N*N_comp]
        us -> us - deltas
        """
        deltas = deltas.view(self.N_component, self.N_us_grad).T
        self._us[self.grad_mask] -= deltas

    def set_grid(self, new_us):
        """
        Set grid to new values. Used for Jacobian computation.
        """
        self._us[self.grad_mask] = new_us

    def get_us_mask(self):
        """
        Return us, and mask of which elements are trainable. Used for masking Jacobian equations.
        """
        return self._us, self.grad_mask, self.pde_mask

    def get_real_us_Xs(self):
        """ Return all actual grid points, excluding fake boundaries. """
        return self._us[self.u_mask], self._Xs[self.u_mask]

    def get_all_us_Xs(self):
        """ Return all grid points, including fake boundaries. """
        return self._us, self._Xs

    def get_us_grad(self):
        """ Return us with gradients. """
        return self._us[self.grad_mask]

    def get_us_Xs_pde(self):
        """ Return us and Xs for PDE points. """
        return self._us[self.pde_mask], self._Xs[self.pde_mask]

    def add_nograd_to_us(self, us_grad):
        """
        Add points that don't have gradient to us. Used for Jacobian computation.
        """

        us_all = torch.clone(self._us)
        us_all[self.grad_mask] = us_grad

        return us_all


    @abstractmethod
    def _cuda(self):
        pass