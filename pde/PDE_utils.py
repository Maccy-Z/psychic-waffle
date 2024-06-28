import torch
import abc
import math

from DiscreteDerivative import DerivativeCalc1D
from U_grid import UGrid1D, UGridOpen1D, UGridClosed1D
from X_grid import XGrid


class PDEHandler(abc.ABC):
    @abc.abstractmethod
    def residuals(self, us_bc: torch.Tensor, extra_args):
        """
        PDE function that returns residuals.
        First parameter is tensor which is used for Jacobian.
        extra_args contains extra conditioning for PDE.
        Returns a tuple of identical tensors for jacfwd to calculate jacobian and get value.
        """
        pass


class PDE_forward(PDEHandler):
    def __init__(self, u_grid: UGrid1D):
        self.u_grid = u_grid
        self.derivative_calculator = DerivativeCalc1D(self.u_grid.dx, order=2, device=u_grid.device)

        self.thetas = torch.tensor([0.1, 0.5], dtype=torch.float32, device=u_grid.device)

    def forward(self, Us: torch.Tensor, Xs, thetas=None):
        """
        Xs.shape = [n_dim, N]
        Us.shape = [3, N]
        Get PDE to solve: PDE(Us) = PDE(u, du/dx, d2u/dx2) = 0
        """
        if thetas is None:
            thetas = self.thetas

        u, dudx, d2udx2 = Us
        residuals = thetas[0] * d2udx2 + thetas[1] * dudx + torch.sin(u)
        return residuals

    def residuals(self, us_bc, extras):
        """ Given a grid of points:
                Set Dirichlet boundary on u
                Set Neumann boundary on derivatives
                Compute derivatives
                Compute residuals
        """
        Xs = self.u_grid.xs

        us = self.u_grid.remove_bc(us_bc)
        dudx, d2udx2 = self.derivative_calculator.derivative(us_bc)

        Us = torch.stack([us, dudx, d2udx2], dim=0)
        residuals = self.forward(Us, Xs)
        return residuals, residuals

    def get_dfs(self):
        """
        Returns dfdu, dfdu_x, dfdu_xx, dfdtheta
        """
        # Calculate u_x, u_xx
        us_bc = self.u_grid.get_with_bc()  # Shape = [N+2]
        us = self.u_grid.remove_bc(us_bc)  # Shape = [N]
        dudx, d2udx2 = self.derivative_calculator.derivative(us_bc)  # Shape = [N]
        Us = torch.stack([us, dudx, d2udx2], dim=0).requires_grad_(True)

        # Calculate dfdu, dfdu_x, dfdu_xx
        residuals = self.forward(Us, None)
        residuals.backward(gradient=torch.ones_like(residuals))

        dfdU = Us.grad  # dfdu, dfdu_x, dfdu_xx, Shape = [3, N]#

        # Get dfdtheta using Jacobian
        with torch.no_grad():
            dfdtheta = torch.func.jacrev(self.forward, argnums=2)(Us, None, self.thetas)
        return dfdU, dfdtheta


class PDE_adjoint(PDEHandler):
    def __init__(self, u_grid: UGrid1D, a_grid: UGridClosed1D):
        self.u_grid = u_grid
        self.a_grid = a_grid
        self.derivative_calculator = DerivativeCalc1D(self.u_grid.dx, order=2, device=u_grid.device)

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
