import torch
from matplotlib import pyplot as plt
import math
import scipy.sparse.linalg as linalg
import scipy.sparse as sparse
import scipy
import numpy as np
from abc import ABC, abstractmethod
import time

from U_grid import UGrid, USubgrid
from X_grid import XGrid
from PDE_utils import PDEHandler
from utils import show_grid, get_split_indices


class Solver(ABC):
    """
    Finite difference PDE solver
    Given a grid of points, solve the PDE at each point with discrete derivatives.
    This amounts to solving in parallel the equations:
        PDE(u_i-1, u_i, u_i+1) = 0 for i = 1, ..., N
    """

    def __init__(self, pde_func: PDEHandler, sol_grid: UGrid, device='cpu'):
        self.pde_func = pde_func
        self.sol_grid = sol_grid

        self.device = device

    def plot(self, title=None):
        us, xs = self.sol_grid.get_real_u_x()
        plt.plot(xs.cpu(), us.cpu().numpy())
        if title is not None:
            plt.title(title)
        plt.show()

    @abstractmethod
    def find_pde_root(self, extra=None):
        pass

    def solve_linear(self, A: torch.Tensor, b: torch.Tensor):
        """ Solve Ax = b for x """
        # Use sparse solve on cpu, or dense solve on gpu
        # Newton Raphson root finding
        if self.device == 'cuda':
            deltas = torch.linalg.solve(A, b)
        else:
            A = A.numpy()
            b = b.numpy()
            #deltas = linalg.spsolve(A, b, use_umfpack=False)

            A = sparse.csr_matrix(A)
            A = linalg.aslinearoperator(A)
            deltas = linalg.cg(A, b)[0]

            deltas = torch.tensor(deltas)

        return deltas


class SolverNewton(Solver):
    def __init__(self, pde_func: PDEHandler, sol_grid: UGrid, N_iter: int, lr=1., acc: float = 1e-4):
        super().__init__(pde_func, sol_grid, sol_grid.device)
        self.N_iter = N_iter
        self.lr = lr
        self.solve_acc = acc
        self.N_points = sol_grid.N.prod()

    @torch.no_grad()
    def find_pde_root(self, extra=None):
        """
        Find the root of the PDE using Newton Raphson.
        :param extra: Additional conditioning for the PDE
        """
        for i in range(self.N_iter):
            st = time.time()

            us, us_grad_mask, pde_mask = self.sol_grid.get_us_mask()
            pde_mask = pde_mask[1:-1, 1:-1]
            extra = (us_grad_mask, pde_mask)

            us_grad = us[us_grad_mask]
            jacobian, residuals = torch.func.jacfwd(self.pde_func.residuals, has_aux=True, argnums=0)(us_grad, extra)  # N equations, N+2 Us, jacob.shape: [N^d, N^d]

            print(f"Time to calculate jacobian: {time.time() - st:.4g}")
            st = time.time()

            deltas = self.solve_linear(jacobian, residuals)

            print(f'Time to solve: {time.time() - st:.4g}')
            self.sol_grid.update_grid(deltas)

            # print(f'Residual: {torch.mean(torch.abs(residuals)):.3g}, iteration {i}')
            # if torch.mean(torch.abs(residuals)) < self.solve_acc:
            #     break
        # print("Final values:", self.sol_grid.get_with_bc().cpu())


class SolverNewtonSplit(Solver):
    def __init__(self, pde_func: PDEHandler, sol_grid: UGrid, N_iter: int, lr=1., acc: float = 1e-4):
        super().__init__(pde_func, sol_grid, sol_grid.device)
        self.N_iter = N_iter
        self.lr = lr
        self.solve_acc = acc
        self.N_points = sol_grid.N.prod()

    @torch.no_grad()
    def find_pde_root(self, extra=None):
        """
        Find the root of the PDE using Newton Raphson.
        :param extra: Additional conditioning for the PDE
        """

        # Sparsity pattern of Jacobian
        jacob_shape = self.sol_grid.N_us_train  # == len(torch.nonzero(us_grad_mask))
        num_blocks = 48
        split_idxs = get_split_indices(jacob_shape, num_blocks)
        block_size = jacob_shape // num_blocks
        us_stride = self.sol_grid.N[1]  # Stide of us in grid. Second element of N is the number of points in x direction. Overestimate.

        for i in range(self.N_iter):
            print()
            st = time.time()

            us, us_grad_mask, pde_mask = self.sol_grid.get_us_mask()

            jacobian = torch.zeros((jacob_shape, jacob_shape), device=self.device)
            residuals = torch.zeros(jacob_shape, device=self.device)

            # Build up Jacobian from sections of PDE and us for efficiency.
            for xmin, xmax in split_idxs:
                # Diagonal blocks of Jacobian
                ymin = torch.clip(xmin - us_stride, 0, jacob_shape)
                ymax = torch.clip(xmin + block_size + us_stride, 0, jacob_shape)
                pde_slice = slice(xmin, xmax)
                us_slice = slice(ymin, ymax)

                # Subset of pde equations
                pde_true_idx = torch.nonzero(pde_mask)
                pde_true_idx = pde_true_idx[pde_slice]

                pde_submask = torch.zeros_like(pde_mask)
                pde_submask[pde_true_idx[:, 0], pde_true_idx[:, 1]] = True

                # Subset of us
                us_grad_idx = torch.nonzero(us_grad_mask)
                want_us_idx = us_grad_idx[us_slice]

                us_grad_submask = torch.zeros_like(us_grad_mask)
                us_grad_submask[want_us_idx[:, 0], want_us_idx[:, 1]] = True

                # Further clip region PDE is calculated to around pde_mask and us_grad_mask to avoid unnecessary calculations
                nonzero_idx = torch.nonzero(us_grad_submask)
                a, b = torch.min(nonzero_idx, dim=0)[0], torch.max(nonzero_idx, dim=0)[0]
                a, b = a - 1, b + 1  # Add one point of padding. This *should* always be enough depending on bc and pde_mask
                a, b = torch.clamp(a, min=0), torch.clamp(b, min=0)

                us_region_mask = (slice(a[0], b[0] + 1), slice(a[1], b[1] + 1))

                subgrid = USubgrid(self.sol_grid, us_region_mask, us_grad_submask, pde_submask)

                # Get Jacobian
                us_grad = subgrid.get_us_grad()     # us[region][grad_mask]
                jacob, resid = torch.func.jacfwd(self.pde_func.residuals, has_aux=True, argnums=0)(us_grad, subgrid)  # N equations, N+2 Us, jacob.shape: [N^d, N^d]

                # Fill in calculated parts
                jacobian[pde_slice, us_slice] = jacob
                residuals[pde_slice] = resid.flatten()

            print(f"Time to calculate jacobian: {time.time() - st:.4g}")
            st = time.time()

            deltas = self.solve_linear(jacobian, residuals)

            print(f'Time to solve: {time.time() - st:.4g}')

            self.sol_grid.update_grid(deltas)


def main():
    raise NotImplementedError

if __name__ == "__main__":
    main()
