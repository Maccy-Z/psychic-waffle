import torch
import torch.func as func
from codetiming import Timer
import logging

from pde.graph_grid.U_graph import UGraph
from pde.pdes.PDEs import PDEFunc
from pde.loss import Loss
from pde.BasePDEGrad import PDEFwdBase

class PDEForward(PDEFwdBase):
    def __init__(self, u_graph: UGraph, pde_func: PDEFunc):
        self.pde_func = pde_func
        self.u_graph = u_graph


    def residuals(self, u_dus: torch.Tensor, Xs: torch.Tensor):
        residuals = self.pde_func(u_dus, Xs)
        return residuals


class PDEAdjoint:
    def __init__(self, u_graph: UGraph, pde_func: PDEFunc, adj_jacob_calc, adj_lin_solver, loss_fn: Loss):
        self.pde_func = pde_func
        self.u_graph = u_graph
        self.adj_jacob_calc = adj_jacob_calc
        self.adj_lin_solver = adj_lin_solver
        self.loss_fn = loss_fn

        self.DEVICE = u_graph.device

    def adjoint_solve(self):
        """ Solve for adjoint.
            dgdU = J^T * adjoint
         """
        jac_T = self.adj_jacob_calc.jacob_transpose()    # Shape = [N_eq, N_us]

        # One adjoint value for each trained u value, including boundary points.
        us, grad_mask, _ = self.u_graph.get_us_mask()
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
        us_all = self.u_graph.us  # Shape = [N_total].
        Xs = self.u_graph.Xs[self.u_graph.pde_mask]  # Shape = [N_total, 2].

        # 1) Calculate u_x, u_xx
        grads_dict = self.u_graph.deriv_calc.derivative(us_all)  # Shape = [N_derivs][N_pde].
        u_dus = torch.stack(list(grads_dict.values())).T  # shape = [N_pde, N_derivs]

        # Computes adjoint * dfdp as vector jacobian product.
        residuals = self.pde_func(u_dus, Xs)
        residuals.backward(- adjoint)
        return residuals

