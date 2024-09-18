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

from pde.config import FwdConfig
from pde.U_grid import UGrid
from pde.X_grid import XGrid
from pde.pdes.PDE_utils import PDEHandler
from pde.solvers.jacobian import JacobCalc
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


class SolverNewton:
    def __init__(self, pde_func: PDEHandler, sol_grid: UGrid, lin_solver: LinearSolver, jac_calc: JacobCalc, cfg: FwdConfig):
        self.pde_func = pde_func
        self.sol_grid = sol_grid
        self.lin_solver = lin_solver

        self.N_iter = cfg.N_iter
        self.lr = cfg.lr
        self.N_points = sol_grid.N.prod()
        self.solve_acc = cfg.acc

        self.jac_calc = jac_calc
        self.device = sol_grid.device

    @torch.no_grad()
    def find_pde_root(self):
        """
        Find the root of the PDE using Newton Raphson:
            grad(F(x_n)) * (x_{n+1} - x_n) = -F(x_n)

        :param extra: Additional conditioning for the PDE
        """
        for i in range(self.N_iter):
            print()

            with Timer(text="Time to calculate jacobian: : {:.4f}"):
                jacobian, residuals = self.jac_calc.jacobian()

            with Timer(text="Time to solve: : {:.4f}"):
                deltas = self.lin_solver.solve(jacobian, residuals) # self.solve_linear(jacobian, residuals)

            self.sol_grid.update_grid(deltas)

            if self.terminate(residuals, i):
                print("Converged")

    def terminate(self, residuals, i):
        c_print(f'Residual: {torch.mean(torch.abs(residuals)):.3g}, iteration {i}', color='cyan')

        return torch.mean(torch.abs(residuals)) < self.solve_acc
