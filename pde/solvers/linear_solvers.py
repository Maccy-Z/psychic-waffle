import torch
import scipy.sparse.linalg as linalg
from codetiming import Timer

import cupy as cp
import cupyx.scipy.sparse as sp
import cupyx.scipy.sparse.linalg as sp_linalg
from cprint import c_print
from typing import Literal, Callable

from pde.solvers.gmres import gmres
from pde.solvers.pyamgx_holder import PyAMGXManager

SolvStr = Literal['sparse', 'dense', 'iterative', 'amgx']
JacStr = Literal['dense', 'split']

class LinearSolver:
    """ Solve Ax = b for x """
    solver: Callable
    preproc: Callable

    cfg: dict = None
    def __init__(self, mode: SolvStr, device: str, cfg: dict=None):
        self.preproc = self.preproc_default

        if device == "cuda":
            if mode == "dense":
                self.solver = self.cuda_dense
            elif mode == "sparse":
                self.solver = self.cuda_sparse
            elif mode == "iterative":
                self.solver = self.cuda_iterative
                self.cfg = cfg
                self.preproc = self.preproc_sparse
            elif mode == "amgx":
                self.cfg = cfg
                self.amgx_solver = PyAMGXManager().create_solver(cfg)
                self.solver = self.cuda_amgx
                self.preproc = self.preproc_sparse

        elif device == "cpu":
            if mode == "dense":
                self.solver = self.cpu_dense
            elif mode == "sparse":
                self.solver = self.cpu_sparse

    def solve(self, A: torch.Tensor, b: torch.Tensor):
        return self.solver(A, b)

    def cuda_amgx(self, A_cp: cp.array, b: torch.Tensor):
        # Cupy to sparse is faster than torch to sparse
        self.amgx_solver.init_solver_cp(A_cp)

        x = torch.zeros_like(b)
        x = self.amgx_solver.solve(b, x)
        return x

    def cuda_iterative(self, A_cp: cp.array, b: torch.Tensor):
        b_cp = cp.from_dlpack(b)

        # Convert the dense matrix A_cupy to a sparse CSR matrix
        A_sparse_cupy = sp.csr_matrix(A_cp)

        # Solve the sparse linear system Ax = b using CuPy
        x, info = gmres(A_sparse_cupy, b_cp, **self.cfg)
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

    def preproc_tensor(self, A: torch.Tensor):
        """ Preprocess A matrix before solving, convert to sparse if needed so original can be deleted. """
        return self.preproc(A)

    def preproc_default(self, A: torch.Tensor):
        return A

    def preproc_sparse(self, A: torch.Tensor):
        """ Function to convert a torch sparse tensor to a cupy sparse tensor """
        A_cupy = cp.from_dlpack(A)
        A_sparse_cupy = sp.csr_matrix(A_cupy)
        return A_sparse_cupy





