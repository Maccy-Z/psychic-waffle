from U_grid import UGrid1D
from X_grid import XGrid
from PDE_functions import PDE_func
import torch
from matplotlib import pyplot as plt
import math
import time
import scipy.sparse.linalg as linalg

torch.set_printoptions(linewidth=150, precision=3)


class PDESolver:
    """
    Finite difference PDE solver
    Given a grid of points, solve the PDE at each point with discrete derivatives.
    This amounts to solving in parallel the equations:
        PDE(u_i-1, u_i, u_i+1) = 0 for i = 1, ..., N
    """

    def __init__(self, pde_func: PDE_func, sol_grid: UGrid1D, N_iter: int, lr=1., device='cpu'):
        self.pde_func = pde_func
        self.N_iter = N_iter
        self.sol_grid = sol_grid
        self.lr = lr

        self.device = device
        self.solve_acc = 1e-4

    def train_newton(self, extra=None):
        """
        Find the root of the PDE using Newton Raphson.
        :param extra: Additional conditioning for the PDE
        """
        for i in range(self.N_iter):
            us_bc = self.sol_grid.get_with_bc()

            with torch.no_grad():
                jacob, residuals = torch.func.jacfwd(self.pde_func.residuals, has_aux=True, argnums=0)(us_bc, extra)  # N equations, N+2 Us, Shape: [N, N + 2]

                jacob = jacob[:, self.sol_grid.grad_mask]  # Remove boundary points, Shape: [N, N]
                # print(jacob)
                jacob *= self.lr  # Scale to make jacobian better conditioned

                # Use sparse solve on cpu, or dense solve on gpu
                # Newton Raphson root finding
                if self.device == 'cuda':
                    deltas = torch.linalg.solve(jacob, residuals)
                else:
                    jacob = jacob.numpy()
                    deltas = linalg.spsolve(jacob, residuals)
                    deltas = torch.tensor(deltas)

                self.sol_grid.update_grid(deltas)

                if torch.mean(torch.abs(residuals)) < self.solve_acc:
                    print(i)
                    break

        # print("Final values:", self.sol_grid.get_with_bc().cpu())

    def plot(self, title=None):
        us, xs = self.sol_grid.get_real_u_x()
        plt.plot(xs.cpu(), us.cpu().numpy())
        if title is not None:
            plt.title(title)
        plt.show()


def main():
    Xmin, Xmax = 0, 2 * math.pi
    N = 20
    X_grid = XGrid(Xmin, Xmax, N, device='cuda')

    print(X_grid)


if __name__ == "__main__":
    main()
