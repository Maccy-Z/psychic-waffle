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
from pde.solvers.gmres import gmres

SolvStr = Literal['sparse', 'dense', 'iterative']
JacStr = Literal['dense', 'split']

class LinearSolver:
    """ Solve Ax = b for x """
    solver: Callable
    cfg: dict = None
    def __init__(self, mode: SolvStr, device: str, cfg: dict=None):
        if device == "cuda":
            if mode == "dense":
                self.solver = self.cuda_dense
            elif mode == "sparse":
                self.solver = self.cuda_sparse
            elif mode == "iterative":
                self.solver = self.cuda_iterative
                self.cfg = cfg

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
        default_args = {'maxiter': 1000, 'restart': 125, 'rtol': 1e-3}
        x, info = gmres(A_sparse_cupy, b_cupy, **{**default_args, **self.cfg})
        # print(info)
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

    def __init__(self, pde_func: PDEHandler, sol_grid: UGrid, lin_solver: LinearSolver, acc, device='cpu'):
        self.pde_func = pde_func
        self.sol_grid = sol_grid
        self.solve_acc = acc
        self.solver = lin_solver

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
    def __init__(self, pde_func: PDEHandler, sol_grid: UGrid, solver: LinearSolver, jac_mode: JacStr, N_iter: int, lr=1., acc: float = 1e-4):
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
            print()

            with Timer(text="Time to calculate jacobian: : {:.4f}"):
                jacobian, residuals = self.jac_fn.jacobian()

            with Timer(text="Time to solve: : {:.4f}"):
                deltas = self.solve_linear(jacobian, residuals)

            self.sol_grid.update_grid(deltas)

            if self.terminate(residuals, i):
                print("Converged")


