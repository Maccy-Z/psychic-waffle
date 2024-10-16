import torch
import torch.func as func
from codetiming import Timer
import logging

from pde.graph_grid.U_graph import UGraph
from pde.pdes.PDEs import PDEFunc
from pde.loss import Loss
from pde.findiff.fin_deriv_calc import FinDerivCalc
from pde.BaseU import UBase
from sparse_tensor import CSRSummer, CSRRowMultiplier

class PDEForward:
    def __init__(self, u_graph: UGraph, pde_func: PDEFunc, deriv_calc: FinDerivCalc):
        self.pde_func = pde_func
        self.u_graph = u_graph
        self.deriv_calc = deriv_calc

        self.csr_summer = CSRSummer(self.deriv_calc.jacobian())
        self.row_multipliers = [CSRRowMultiplier(spm) for spm in self.deriv_calc.jacobian()]

    def jac_block(self, u_grid: UBase):
        """
            Compute jacobian dR/dU = dR/dD * dD/dU.
            df_i/dU_j = sum_k df_i/dD_k * dD_ik/dU_j

            us_grad.shape = [N_u_grad]. Gradients of trained u values.
            dR/dU.shape = [N_pde, N_u_grad]
            dR/dD.shape = [N_pde, N_derivs]
            dD/dU.shape = [N_pde, N_derivs, N_u_grad]
        """

        us_all = u_grid.us  # Shape = [N_total]. Need all Us to calculate derivatives.
        Xs = u_grid.Xs[u_grid.pde_mask]  # Shape = [N_total, 2]. Only need Xs for residuals.

        # 1) Finite differences D. shape = [N_pde, N_derivs]
        grads_dict = self.deriv_calc.derivative(us_all)  # shape = [N_pde]. Derivative removes boundary points.
        u_dus = torch.stack(list(grads_dict.values())).T    # shape = [N_pde, N_derivs]

        # 2) dD/dU. shape = [N_derivs, N_u_grad]
        # dDdU = torch.autograd.functional.jacobian(temp_fn, us_all)
        dDdU = self.deriv_calc.jacobian() # shape = [N_derivs][N_pde, N_total]

        # 3) Compute dR/dD. shape = [N_pde, N_derivs]
        resid_grad = func.grad_and_value(self.pde_func, argnums=0)
        dRdD, residuals = func.vmap(resid_grad)(u_dus, Xs)

        # 4.1) Take product over i: df_i/dD_k * dD_ik/dU_j. shape = [N_pde, N_u_grad]
        partials = []
        for d in range(len(dDdU)):
            # prod = torch.mul(dDdU[d], dRdD[:, d].unsqueeze(-1).repeat(1, u_grid.N_us_grad))
            # partials.append(prod)
            prod = self.row_multipliers[d].mul(dDdU[d], dRdD[:, d])
            partials.append(prod)

        # 4.2) Sum over k: sum_k partials_ijk
        jacobian = self.csr_summer.sum(partials)
            # jacobian = partials[0]
            # for part in partials[1:]:
            #     jacobian += part

        return jacobian.to_dense(), residuals


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
