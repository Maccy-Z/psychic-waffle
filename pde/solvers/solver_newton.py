from codetiming import Timer
import logging
import torch

from pde.config import FwdConfig
from pde.BaseU import UBase
from pde.cartesian_grid.PDE_Grad import PDEForward
from pde.solvers.jacobian import JacobCalc
from pde.solvers.linear_solvers import LinearSolver

class SolverNewton:
    def __init__(self, pde_func: PDEForward, sol_grid: UBase, lin_solver: LinearSolver, jac_calc: JacobCalc, cfg: FwdConfig):
        self.pde_func = pde_func
        self.sol_grid = sol_grid
        self.lin_solver = lin_solver

        self.N_iter = cfg.N_iter
        self.lr = cfg.lr
        # self.N_points = sol_grid.N_points
        self.solve_acc = cfg.acc

        self.jac_calc = jac_calc
        self.device = sol_grid.device

    def find_pde_root(self):
        """
        Find the root of the PDE using Newton Raphson:
            grad(F(x_n)) * (x_{n+1} - x_n) = -F(x_n)

        :param extra: Additional conditioning for the PDE
        """
        for i in range(self.N_iter):
            logging.debug("\n")
            with Timer(text="Time to calculate jacobian: : {:.4f}", logger=logging.debug):
                jacobian, residuals = self.jac_calc.jacobian()

            # from pde.utils_sparse import plot_sparsity
            # print(f'{jacobian.shape = }')
            # plot_sparsity(jacobian)
            # jacobian = jacobian.to_dense()
            # print(f'rank = {torch.linalg.matrix_rank(jacobian)}')
            #
            # print(torch.min(torch.inverse(jacobian)))
            # exit(4)

            with Timer(text="Time to solve: : {:.4f}", logger=logging.debug):
                # Convert jacobian to sparse here instead of in lin_solver, so we can delete the dense Jacobian asap.
                jac_preproc = self.lin_solver.preproc_tensor(jacobian)
                del jacobian
                # torch.cuda.empty_cache()
                deltas = self.lin_solver.solve(jac_preproc, residuals)

            deltas *= self.lr
            self.sol_grid.update_grid(deltas)

            logging.debug(f'Iteration {i}, Mean residual: {torch.mean(torch.abs(residuals)):.3g}')
            if torch.mean(torch.abs(residuals)) < self.solve_acc:
                logging.info(f"Newton solver converged early at iteration {i+1}")
                break

