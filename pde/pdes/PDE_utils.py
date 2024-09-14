import torch
from torch.func import functional_call, vmap, grad

import abc
import math

from pde.pdes.discrete_derivative import DerivativeCalc
from pde.U_grid import UGrid2D, UGrid, USplitGrid, USubGrid
from pde.X_grid import XGrid
from .PDEs import PDEFunc


class PDEHandler(abc.ABC):
    pde_func: PDEFunc

    def __init__(self, pde_func: PDEFunc):
        self.pde_func = pde_func

    @abc.abstractmethod
    def residuals(self, us_bc: torch.Tensor, extra_args) -> torch.Tensor:
        """
            PDE function that returns residuals.
            First parameter is tensor which is used for Jacobian.
            extra_args contains extra conditioning for PDE.
            Returns a tuple of identical tensors for jacfwd to calculate jacobian and get value.
        """
        pass


class PDEForward(PDEHandler):
    us_dus_cache: torch.Tensor

    def __init__(self, u_grid: UGrid, pde_func: PDEFunc, deriv_calc: DerivativeCalc):
        super().__init__(pde_func)
        self.u_grid = u_grid
        self.deriv_calc = deriv_calc

    def residuals(self, us_grad, subgrid: USubGrid):
        """
            Returns residuals of equations that require gradients only.
        """

        us_bc = subgrid.add_nograd_to_us(us_grad)  # Shape = [N+2, ...]. Need all Us to calculate derivatives.
        Xs = subgrid.Xs_pde  # Shape = [N, ...]. Only need Xs for residuals.
        us = us_bc[1:-1, 1:-1].unsqueeze(0)

        dudX, d2udX2 = self.deriv_calc.derivative(us_bc)  # shape = [N, ...]. Derivative removes boundary points.

        us_dus = (us, dudX, d2udX2)
        residuals = self.pde_func.residuals(us_dus, Xs)
        resid_grad = residuals[subgrid.pde_mask]

        return resid_grad, resid_grad

    def only_resid(self):
        us_bc, Xs_bc = self.u_grid.get_all_us_Xs()
        dudX, d2udX2 = self.deriv_calc.derivative(us_bc)
        us, _ = self.u_grid.get_real_us_Xs()

        us_dus = (us, dudX, d2udX2)
        residuals = self.pde_func.residuals(us_dus, Xs_bc)

        return residuals

    def get_dfs(self):
        """
        Gets gradients of f at current state U
        Returns: [dfdu, dfdu_x, dfdu_xx]
                  dfdtheta: dict of df/dtheta at each point, for each param group
        """
        # 1) Calculate u_x, u_xx
        N = self.u_grid.N
        us_bc, _ = self.u_grid.get_all_us_Xs()  # Shape = [N+2]
        us, Xs = self.u_grid.get_real_us_Xs()  # Shape = [N]
        dudx, d2udx2 = self.deriv_calc.derivative(us_bc)  # Shape = [N]
        # Mask for boundary points
        mask = self.u_grid.pde_mask[1:-1, 1:-1]

        # 2) Reshape Us for vmap
        us = us.unsqueeze(0)
        Xs = Xs.permute(2, 0, 1)
        dUs = [us, dudx, d2udx2]
        dUs_flat = [torch.flatten(u.permute(1, 2, 0), start_dim=0, end_dim=1) for u in dUs]
        Xs_flat = torch.flatten(Xs.permute(1, 2, 0), start_dim=0, end_dim=1)

        # 3) Get dfdtheta using vmap
        params = {k: v.detach() for k, v in self.pde_func.named_parameters()}
        buffers = {k: v.detach() for k, v in self.pde_func.named_buffers()}

        def get_output(_params, _buffers, _dU, _X):
            _dU = [du.unsqueeze(-1).unsqueeze(-1) for du in _dU]        # pde_func expects [Nitem, Nx, Ny]
            _X = _X.unsqueeze(-1).unsqueeze(-1)
            ret = functional_call(self.pde_func, (_params, _buffers), (_dU, _X))
            return ret[0, 0]

        all_grad_fn = vmap(grad(get_output), in_dims=(None, None, 0, 0), out_dims=0)

        dfdtheta = all_grad_fn(params, buffers, dUs_flat, Xs_flat)

        # Reshape to square array and mask off boundary points
        for name, param in dfdtheta.items():
            n_params = param.shape[-1]
            param = param.view([*N, n_params])
            param = param * mask[..., None]
            dfdtheta[name] = param

        return dfdtheta


class PDE_adjoint(PDEHandler):
    def __init__(self, u_grid: UGrid, a_grid: UGrid, pde_func: PDEFunc, deriv_calc: DerivativeCalc):
        self.u_grid = u_grid
        self.a_grid = a_grid
        self.deriv_calc = deriv_calc

    def residuals(self, as_bc: torch.Tensor, Lu_dfdU):
        """
        Solve adjoint equation for a
        Requires dfdU = [dF/du, dF/du_x, dF/du_xx]
        Returns adjoint_PDE(a) = a F_u - (a F_u_x)_x + (a F_u_xx)_xx - L_u
        """

        Lu, dfdU = Lu_dfdU  # Shape = [N-2], [3, N]

        a = self.a_grid.remove_bc(as_bc)  # Shape = [N]

        a_dfdU = a * dfdU  # Shape = [3, N]
        af_u = a_dfdU[0]  # Shape = [N]
        af_u_x = a_dfdU[1]  # Shape = [N]
        af_u_xx = a_dfdU[2]  # Shape = [N]

        daf_u_x_dx, _ = self.derivative_calculator.derivative_boundary(af_u_x)  # Shape = [N]
        _, d2af_u_xx_dx2 = self.derivative_calculator.derivative_boundary(af_u_xx)  # Shape = [N]

        adjoint_resid = af_u - daf_u_x_dx + d2af_u_xx_dx2

        # Remove first and last term on boundary since equations are overdetermined
        adjoint_resid = adjoint_resid[1:-1]

        adjoint_resid = adjoint_resid + Lu
        return adjoint_resid, adjoint_resid


class Loss_fn:
    def __init__(self, X_grid: XGrid, true_us: torch.Tensor):
        """
        :param xs: Positions where loss is taken at
        :param true_us: Value of us at xs
        Note: Grid is assumed to be fixed

        Returns: dL(u)/du for adjoint equation
        """

        self.true_us = true_us  # Shape = [N-2]
        self.xs = X_grid

    def loss_fn(self, us):
        loss = torch.sum((us - self.true_us) ** 2)  # Shape = [N-2]
        return loss

    def general_loss(self, us):
        us.requires_grad_(True)
        loss = self.loss_fn(us)
        loss.backward()
        return us.grad

    def quad_loss(self, us: torch.Tensor):
        # Special case for quadratic loss
        # loss = torch.sum((us - self.true_us)** 2)
        dL_du = 2 * (us - self.true_us)

        return dL_du


def main():
    pass


if __name__ == "__main__":
    main()
