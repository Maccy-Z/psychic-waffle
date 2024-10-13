import torch
from codetiming import Timer
import logging

from pde.graph_grid.U_graph import UGraph
from pde.pdes.PDEs import PDEFunc
from pde.loss import Loss
from pde.findiff.fin_deriv_calc import FinDerivCalc

class PDEForward:
    def __init__(self, u_graph: UGraph, pde_func: PDEFunc, deriv_calc: FinDerivCalc):
        self.pde_func = pde_func
        self.u_graph = u_graph
        self.deriv_calc = deriv_calc

    def residuals(self, us_grad, subgrid):
        """
            Returns residuals of equations that require gradients only.
        """
        us_all = subgrid.add_nograd_to_us(us_grad)  # Shape = [N_total]. Need all Us to calculate derivatives.
        Xs = subgrid.Xs_pde  # Shape = [N_total, 2]. Only need Xs for residuals.

        us_pde = us_all[subgrid.pde_mask]  # Shape = [N, ...]. Only need Us for residuals.

        grads_dict = self.deriv_calc.derivative(us_all)  # shape = [N, ...]. Derivative removes boundary points.
        grads_dict[(0, 0)] = us_pde

        residuals = self.pde_func.residuals(grads_dict, Xs)
        resid_grad = residuals[subgrid.pde_mask]

        return resid_grad, resid_grad

    def only_resid(self):
        """ Only returns residuals. Used for tracking solve progress."""
        us_bc, Xs_bc = self.u_grid.get_all_us_Xs()
        dudX, d2udX2 = self.deriv_calc.derivative(us_bc)
        us, _ = self.u_grid.get_real_us_Xs()

        us_dus = (us, dudX, d2udX2)
        residuals = self.pde_func.residuals(us_dus, Xs_bc)

        return residuals


class PDEAdjoint:
    def __init__(self, u_graph: UGraph, pde_func: PDEFunc, adj_jacob_calc, adj_lin_solver, loss_fn: Loss):
        self.pde_func = pde_func
        self.us_grid = u_graph
        self.adj_jacob_calc = adj_jacob_calc
        self.adj_lin_solver = adj_lin_solver
        self.loss_fn = loss_fn

        self.DEVICE = u_graph.device

    def adjoint_solve(self):
        """ Solve for adjoint.
            dgdU = J^T * adjoint
         """
        with torch.no_grad():
            jacobian, _ = self.adj_jacob_calc.jacobian()    # Shape = [N_eq, N_us]
            jac_T = jacobian.T
            del jacobian

        # One adjoint value for each trained u value, including boundary points.
        us, grad_mask, _ = self.us_grid.get_us_mask()
        us_grad = us[grad_mask]

        loss = self.loss_fn(us_grad)
        loss_u = self.loss_fn.gradient()
        with Timer(text="Adjoint solve: {:.4f}", logger=logging.debug):
            # Free memory of dense jacobian before solving adjoint equation.
            jac_T_proc = self.adj_lin_solver.preproc_tensor(jac_T)
            del jac_T
            adjoint = self.adj_lin_solver.solve(jac_T_proc, loss_u)

        return adjoint, loss

    def backpropagate(self, adjoint):
        """
            Computes grads and populates model.parameters.grads
            adjoint.shape = [N_us] = [N_PDEs]
        """

        # 1) Calculate u_x, u_xx
        us_bc, _ = self.us_grid.get_all_us_Xs()  # Shape = [N+2, N+2]
        us, Xs = self.us_grid.get_real_us_Xs()  # Shape = [N, N], [N, N, 2]
        dudx, d2udx2 = self.deriv_calc.derivative(us_bc)  # Shape = [N]
        # Mask for boundary points
        pde_mask = self.us_grid.pde_mask[1:-1, 1:-1]    # Shape = [N, N]

        # 2) Reshape Us for vmap
        us = us[pde_mask].unsqueeze(-1)
        dudx = dudx[pde_mask]
        d2udx2 = d2udx2[pde_mask]
        dUs_flat = [us, dudx, d2udx2]
        Xs_flat = Xs[pde_mask]

        # Computes adjoint * dfdp as vector jacobian product.
        residuals = self.pde_func(dUs_flat, Xs_flat)
        residuals.backward(- adjoint)
        return residuals


def main():
    pass


if __name__ == "__main__":
    main()
