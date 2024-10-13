from abc import ABC, abstractmethod
import torch

class PDEFwdBase(ABC):
    @abstractmethod
    def residuals(self, us_grad: torch.Tensor, subgrid):
        """  Returns residuals of equations that require gradients only. """
        pass

    @abstractmethod
    def only_resid(self):
        """ Only returns residuals. Used for tracking solve progress."""
        pass


class PDEAdjBase(ABC):  # TODO: Probably doesn't need to be abstract.
    @abstractmethod
    def adjoint_solve(self):
        pass
        # """ Solve for adjoint.
        #     dgdU = J^T * adjoint
        #  """
        # with torch.no_grad():
        #     jacobian, _ = self.adj_jacob_calc.jacobian()    # Shape = [N_eq, N_us]
        #     jac_T = jacobian.T
        #     del jacobian
        #
        # # One adjoint value for each trained u value, including boundary points.
        # us, grad_mask, _ = self.us_grid.get_us_mask()
        # us_grad = us[grad_mask]
        #
        # loss = self.loss_fn(us_grad)
        # loss_u = self.loss_fn.gradient()
        # with Timer(text="Adjoint solve: {:.4f}", logger=logging.debug):
        #     # Free memory of dense jacobian before solving adjoint equation.
        #     jac_T_proc = self.adj_lin_solver.preproc_tensor(jac_T)
        #     del jac_T
        #     adjoint = self.adj_lin_solver.solve(jac_T_proc, loss_u)
        #
        # return adjoint, loss

    @abstractmethod
    def backpropagate(self, adjoint):
        pass
        # """
        #     Computes grads and populates model.parameters.grads
        #     adjoint.shape = [N_us] = [N_PDEs]
        # """
        #
        # # 1) Calculate u_x, u_xx
        # us_bc, _ = self.us_grid.get_all_us_Xs()  # Shape = [N+2, N+2]
        # us, Xs = self.us_grid.get_real_us_Xs()  # Shape = [N, N], [N, N, 2]
        # dudx, d2udx2 = self.deriv_calc.derivative(us_bc)  # Shape = [N]
        # # Mask for boundary points
        # pde_mask = self.us_grid.pde_mask[1:-1, 1:-1]    # Shape = [N, N]
        #
        # # 2) Reshape Us for vmap
        # us = us[pde_mask].unsqueeze(-1)
        # dudx = dudx[pde_mask]
        # d2udx2 = d2udx2[pde_mask]
        # dUs_flat = [us, dudx, d2udx2]
        # Xs_flat = Xs[pde_mask]
        #
        # # Computes adjoint * dfdp as vector jacobian product.
        # residuals = self.pde_func(dUs_flat, Xs_flat)
        # residuals.backward(- adjoint)
        # return residuals