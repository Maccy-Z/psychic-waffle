import torch
from matplotlib import pyplot as plt
import math
import scipy.sparse.linalg as linalg
from codetiming import Timer

from abc import ABC, abstractmethod
import time
import cupy as cp
import cupyx.scipy.sparse as sp
import cupyx.scipy.sparse.linalg as sp_linalg
from cprint import c_print
from typing import Literal, Callable

from pde.U_grid import UGrid, UNormalGrid
from pde.X_grid import XGrid
from pde.pdes.PDE_utils import PDEHandler
from pde.utils import show_grid, get_split_indices
from pde.solvers.jacobian import SplitJacobCalc, JacobCalc


SolvStr = Literal['sparse', 'dense', 'iterative']
JacStr = Literal['dense', 'split']

class LinearSolver:
    """ Solve Ax = b for x """
    solver: Callable

    def __init__(self, mode: SolvStr, device: str):
        if device == "cuda":
            if mode == "dense":
                self.solver = self.cuda_dense
            elif mode == "sparse":
                self.solver = self.cuda_sparse
            elif mode == "iterative":
                self.solver = self.cuda_iterative

        elif device == "cpu":
            if mode == "dense":
                self.solver = self.cpu_dense
            elif mode == "sparse":
                self.solver = self.cpu_sparse

    def solve(self, A: torch.Tensor, b: torch.Tensor):
        return self.solver(A, b)

    def cuda_iterative(self, A: torch.Tensor, b: torch.Tensor):
        A_cupy = cp.from_dlpack(A)
        b_cupy = cp.from_dlpack(b)

        # Convert the dense matrix A_cupy to a sparse CSR matrix
        A_sparse_cupy = sp.csr_matrix(A_cupy)

        # Solve the sparse linear system Ax = b using CuPy
        x, info = sp_linalg.gmres(A_sparse_cupy, b_cupy, maxiter=1000, restart=125, tol=0.)
        x = torch.from_dlpack(x)
        return x

    def cuda_sparse(self, A: torch.Tensor, b: torch.Tensor):
        A_cupy = cp.from_dlpack(A)
        b_cupy = cp.from_dlpack(b)

        # Convert the dense matrix A_cupy to a sparse CSR matrix
        A_sparse_cupy = sp.csr_matrix(A_cupy)

        # Solve the sparse linear system Ax = b using CuPy
        x = sp_linalg.spsolve(A_sparse_cupy, b_cupy)

        x = torch.from_dlpack(x)
        return x

    def cuda_dense(self, A: torch.Tensor, b: torch.Tensor):
        deltas = torch.linalg.solve(A, b)
        return deltas

    def cpu_sparse(self, A: torch.Tensor, b: torch.Tensor):
        A = A.numpy()
        b = b.numpy()
        deltas = linalg.spsolve(A, b, use_umfpack=True)
        deltas = torch.from_numpy(deltas)
        return deltas

    def cpu_dense(self, A: torch.Tensor, b: torch.Tensor):
        deltas = torch.linalg.solve(A, b)
        return deltas



class Solver(ABC):
    """
    Finite difference PDE solver
    Given a grid of points, solve the PDE at each point with discrete derivatives.
    This amounts to solving in parallel the equations:
        PDE(u_i-1, u_i, u_i+1) = 0 for i = 1, ..., N
    """

    def __init__(self, pde_func: PDEHandler, sol_grid: UGrid, solver: SolvStr, acc, device='cpu'):
        self.pde_func = pde_func
        self.sol_grid = sol_grid
        self.solve_acc = acc

        self.solver = LinearSolver(solver, device)

        self.device = device

    def plot(self, title=None):
        us, xs = self.sol_grid.get_real_u_x()
        plt.plot(xs.cpu(), us.cpu().numpy())
        if title is not None:
            plt.title(title)
        plt.show()

    def terminate(self, residuals, i):
        c_print(f'Residual: {torch.mean(torch.abs(residuals)):.3g}, iteration {i}', color='cyan')

        return torch.mean(torch.abs(residuals)) < self.solve_acc

    @abstractmethod
    def find_pde_root(self, extra=None):
        pass

    def solve_linear(self, A: torch.Tensor, b: torch.Tensor):
        deltas = self.solver.solve(A, b)
        return deltas


class SolverNewton(Solver):
    def __init__(self, pde_func: PDEHandler, sol_grid: UGrid, solver: SolvStr, jac_mode: JacStr, N_iter: int, lr=1., acc: float = 1e-4):
        super().__init__(pde_func, sol_grid, solver, acc, sol_grid.device)
        self.N_iter = N_iter
        self.lr = lr
        self.N_points = sol_grid.N.prod()

        if jac_mode == "dense":
            self.jac_fn = JacobCalc(sol_grid, pde_func)
        elif jac_mode == "split":
            self.jac_fn = SplitJacobCalc(sol_grid, pde_func, num_blocks=4)
        else:
            raise ValueError(f"Invalid jac_mode {jac_mode}. Must be 'dense' or 'split'")

    @torch.no_grad()
    def find_pde_root(self, extra=None):
        """
        Find the root of the PDE using Newton Raphson.
        :param extra: Additional conditioning for the PDE
        """
        for i in range(self.N_iter):
            with Timer(text="Time to calculate jacobian: : {:.4f}"):
                jacobian, residuals = self.jac_fn.jacobian()

            with Timer(text="Time to solve: : {:.4f}"):
                deltas = self.solve_linear(jacobian, residuals)

            self.sol_grid.update_grid(deltas)

            if self.terminate(residuals, i):
                print("Converged")


# class SolverNewtonSplit(Solver):
#     def __init__(self, pde_func: PDEHandler, sol_grid: UGrid, solver: SolvStr, N_iter: int, lr=1., acc: float = 1e-4):
#         super().__init__(pde_func, sol_grid, solver, acc, sol_grid.device)
#         self.N_iter = N_iter
#         self.lr = lr
#         self.N_points = sol_grid.N.prod()
#
#     @torch.no_grad()
#     def find_pde_root(self, extra=None):
#         """
#         Find the root of the PDE using Newton Raphson.
#         :param extra: Additional conditioning for the PDE
#         """
#
#         # Sparsity pattern of Jacobian
#         jacob_shape = self.sol_grid.N_us_train.item()  # == len(torch.nonzero(us_grad_mask))
#         num_blocks = 2
#         split_idxs = get_split_indices(jacob_shape, num_blocks)
#         block_size = jacob_shape // num_blocks
#         us_stride = self.sol_grid.N[1] +1 # Stide of us in grid. Second element of N is the number of points in x direction. Overestimate.
#
#         for i in range(self.N_iter):
#             print()
#             with Timer(text="Time to calculate jacobian: : {:.4f}"):
#
#                 us, us_grad_mask, pde_mask = self.sol_grid.get_us_mask()
#
#                 jacobian = torch.zeros((jacob_shape, jacob_shape), device=self.device)
#                 residuals = torch.zeros(jacob_shape, device=self.device)
#
#                 # Build up Jacobian from sections of PDE and us for efficiency.
#                 # Rectangular blocks of Jacobian between [xmin, xmax] (pde slice) and [ymin, ymax] (us slice)
#                 for xmin, xmax in split_idxs:
#                     # Diagonal blocks of Jacobian
#                     ymin = torch.clip(xmin - us_stride, 0, jacob_shape)
#                     ymax = torch.clip(xmin + block_size + us_stride, 0, jacob_shape)
#
#                     pde_slice = slice(xmin, xmax)
#                     us_slice = slice(ymin, ymax)
#
#                     # Subset of pde equations
#                     pde_true_idx = torch.nonzero(pde_mask)
#                     pde_true_idx = pde_true_idx[pde_slice]
#
#                     pde_submask = torch.zeros_like(pde_mask)
#                     pde_submask[pde_true_idx[:, 0], pde_true_idx[:, 1]] = True
#
#                     # Subset of us
#                     us_grad_idx = torch.nonzero(us_grad_mask)
#                     want_us_idx = us_grad_idx[us_slice]
#
#                     us_grad_submask = torch.zeros_like(us_grad_mask)
#                     us_grad_submask[want_us_idx[:, 0], want_us_idx[:, 1]] = True
#
#                     # Further clip region PDE is calculated to around pde_mask and us_grad_mask to avoid unnecessary calculations
#                     nonzero_idx = torch.nonzero(us_grad_submask)
#                     a, b = torch.min(nonzero_idx, dim=0)[0], torch.max(nonzero_idx, dim=0)[0]
#                     a, b = a - 1, b + 1  # Add one point of padding. This *should* always be enough depending on bc and pde_mask
#                     a, b = torch.clamp(a, min=0), torch.clamp(b, min=0)
#                     us_region_mask = (slice(a[0], b[0] + 1), slice(a[1], b[1] + 1))
#
#                     subgrid = USplitGrid(self.sol_grid, us_region_mask, us_grad_submask, pde_submask)
#
#
#                     # Get Jacobian
#                     us_grad = subgrid.get_us_grad()  # us[region][grad_mask]
#                     jacob, resid = torch.func.jacfwd(self.pde_func.residuals, has_aux=True, argnums=0)(us_grad, subgrid)  #jacob.shape: [block_size, block_size+stride]
#
#                     # Fill in calculated parts
#                     jacobian[pde_slice, us_slice] = jacob
#                     residuals[pde_slice] = resid.flatten()
#
#             show_grid(jacobian, "Jacobian")
#             with Timer(text="Time to solve: : {:.4f}"):
#                 deltas = self.solve_linear(jacobian, residuals)
#                 deltas *= self.lr
#
#                 self.sol_grid.update_grid(deltas)
#
#             if self.terminate(residuals, i):
#                 print("Converged")


