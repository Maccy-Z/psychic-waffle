import torch
from matplotlib import pyplot as plt
import math
import scipy.sparse.linalg as linalg
import scipy.sparse as sparse
import scipy
import numpy as np
from abc import ABC, abstractmethod
import time

from U_grid import UGrid
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


def create_band_matrix(diagonal_values, off_diagonal_values, n, size):
    """
    Creates a matrix with specified diagonal elements and elements n rows/columns above and below the diagonal.

    Parameters:
    diagonal_values (float): Value to be filled in the main diagonal.
    off_diagonal_values (float): Value to be filled in the diagonals n rows/columns above and below the main diagonal.
    n (int): Number of rows/columns above and below the main diagonal.
    size (int): Size of the square matrix.

    Returns:
    torch.Tensor: The resulting band matrix.
    """
    # Initialize the matrix with zeros
    matrix = torch.zeros(size, size)

    # Fill the main diagonal
    matrix += torch.diag(torch.full((size,), diagonal_values))

    # Fill the off diagonals
    for i in range(1, n + 1):
        matrix += torch.diag(torch.full((size - i,), off_diagonal_values), i)
        matrix += torch.diag(torch.full((size - i,), off_diagonal_values), -i)

    return matrix


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
            us, us_grad_mask, pde_mask = self.sol_grid.get_us_mask()
            pde_mask = pde_mask[1:-1, 1:-1]
            extra = (us_grad_mask, pde_mask)

            us_grad = us[us_grad_mask]
            jacob, residuals = torch.func.jacfwd(self.pde_func.residuals, has_aux=True, argnums=0)(us_grad, extra)  # N equations, N+2 Us, jacob.shape: [N^d, N^d]

            # Flatten grid to vectors
            residuals = residuals.flatten()
            # jacob *= self.lr  # Reduce update scale

            # Use sparse solve on cpu, or dense solve on gpu
            # Newton Raphson root finding
            if self.device == 'cuda':
                deltas = torch.linalg.solve(jacob, residuals)
            else:
                jacob = jacob.numpy()
                # jacob = sparse.csr_matrix(jacob)
                # deltas = linalg.spsolve(jacob, residuals, use_umfpack=False)
                jacob = sparse.csr_matrix(jacob)
                jacob = linalg.aslinearoperator(jacob)
                deltas = linalg.lgmres(jacob, residuals)[0]
                deltas = torch.tensor(deltas)
            self.sol_grid.update_grid(deltas)

            print(f'Residual: {torch.mean(torch.abs(residuals))}, iteration {i}')
            if torch.mean(torch.abs(residuals)) < self.solve_acc:
                break
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
        num_blocks = 8
        split_idxs = get_split_indices(jacob_shape, num_blocks)
        block_size = jacob_shape // num_blocks
        us_stride = self.sol_grid.N[1]  # Stide of us in grid. Second element of N is the number of points in x direction. Overestimate.

        for i in range(self.N_iter):
            us, us_grad_mask, pde_mask = self.sol_grid.get_us_mask()
            pde_mask = pde_mask[1:-1, 1:-1]

            jacobian = torch.zeros((jacob_shape, jacob_shape))

            # Build up Jacobian from sections of PDE and us for efficiency.
            for xmin, xmax in split_idxs:
                # Diagonal blocks of Jacobian
                ymin = np.clip(xmin - us_stride, 0, jacob_shape)
                ymax = np.clip(xmin + block_size + us_stride, 0, jacob_shape)
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

                # Get Jacobian
                masks = (us_grad_submask, pde_submask)
                us_grad = us[us_grad_submask]
                jacob, residuals = torch.func.jacfwd(self.pde_func.residuals, has_aux=True, argnums=0)(us_grad, masks)  # N equations, N+2 Us, jacob.shape: [N^d, N^d]

                jacobian[pde_slice, us_slice] = jacob

            sparsity = jacobian != 0
            show_grid(sparsity, "Split Jacobian sparsity ")
            print(torch.sum(sparsity) / sparsity.numel())


def main():
    Xmin, Xmax = 0, 2 * math.pi
    N = 20
    X_grid = XGrid(Xmin, Xmax, N, device='cuda')

    print(X_grid)


if __name__ == "__main__":
    main()
