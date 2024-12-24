from codetiming import Timer
import time
import logging
import torch

from pde.config import FwdConfig
from pde.BaseU import UBase
from pde.cartesian_grid.PDE_Grad import PDEForward
from pde.solvers.jacobian import JacobCalc
from pde.solvers.linear_solvers import LinearSolver

class SolverNewton:
    def __init__(self,  sol_grid: UBase, lin_solver: LinearSolver, jac_calc: JacobCalc, cfg: FwdConfig):
        #self.pde_func = pde_func
        self.sol_grid = sol_grid
        self.lin_solver = lin_solver

        self.N_iter = cfg.N_iter
        self.lr = cfg.lr
        self.solve_acc = cfg.acc

        self.jac_calc = jac_calc
        self.device = sol_grid.device

        self.logging = self.new_log()

    def find_pde_root(self, aux_input=None):
        """
        Find the root of the PDE using Newton Raphson:
            grad(F(x_n)) * (x_{n+1} - x_n) = -F(x_n)

        :param aux_input: Additional conditioning for the PDE
        """
        for i in range(self.N_iter):
            logging.debug("\n")
            with Timer(text="Time to calculate jacobian: : {:.4f}", logger=logging.debug):
                jacobian, residuals = self.jac_calc.jacobian(aux_input)

            if i == 3:
                torch.save(jacobian, "jacobian.pt")
                torch.save(residuals, "residuals.pt")
                exit(8)
            mean_abs_residual = torch.mean(torch.abs(residuals))
            max_abs_residual = torch.max(torch.abs(residuals))
            print(f'{i} Mean residual: {mean_abs_residual:.4g}, Max residual: {max_abs_residual:.4g}')

            st = time.time()
            with Timer(text="Time to solve: : {:.4f}", logger=logging.debug):
                # Convert jacobian to sparse here instead of in lin_solver, so we can delete the dense Jacobian asap.
                jac_preproc = self.lin_solver.preproc_tensor(jacobian)
                del jacobian # torch.cuda.empty_cache()
                deltas = self.lin_solver.solve(jac_preproc, residuals)

            self.logging["time"] += time.time() - st
            deltas *= self.lr
            self.sol_grid.update_grid(deltas)

            residuals = self.jac_calc.residuals(aux_input)
            mean_abs_residual = torch.mean(torch.abs(residuals))
            max_abs_residual = torch.max(torch.abs(residuals))
            self.logging["residual"] = mean_abs_residual

            logging.debug(f'Iteration {i}, Mean residual: {mean_abs_residual:.4g}, Max residual: {max_abs_residual:.4g}')
            if torch.mean(torch.abs(residuals)) < self.solve_acc:
                logging.info(f"Newton solver converged early at iteration {i+1}")
                break





    def new_log(self):
        return {"time": 0.0, "residual": 0.}