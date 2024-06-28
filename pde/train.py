import torch
import math
from pde_solve import PDESolver
from PDE_functions import PDE_forward, PDE_adjoint, Loss_fn
from U_grid import UGrid1D, UGridOpen1D, UGridClosed #PDEGridOpen, PDEGridClosed, PointGrid
from X_grid import XGrid

class Trainer:
    dfdtheta: torch.Tensor = None

    def __init__(self, device='cpu'):
        self.device = device

        # Init finite difference grids and solutions
        Xmin, Xmax = 0, 2 * math.pi
        N = 100
        X_grid = XGrid(Xmin, Xmax, N, device=device)

        self.X_grid = X_grid
        self.u_grid = UGridOpen1D(X_grid, dirichlet_bc={'x0_lower': 0}, neuman_bc={'x0_lower': 1}, device=device)
        self.a_grid = UGridClosed(X_grid, dirichlet_bc={'x0_upper': 0}, neuman_bc={'x0_upper': 0}, device=device)

        # PDE forward and adjoint functions
        self.pde_fwd = PDE_forward(self.u_grid)
        self.pde_adjoint = PDE_adjoint(self.u_grid, a_grid=self.a_grid)

        # Solver for forward and adjoint pdes
        self.fwd_solver = PDESolver(self.pde_fwd, sol_grid=self.u_grid, N_iter=40, lr=2, device=device)
        self.adj_solver = PDESolver(self.pde_adjoint, sol_grid=self.a_grid, N_iter=4, lr=1, device=device)

        # Loss funtion to fit
        self.loss_fn = Loss_fn(X_grid=X_grid, true_us=torch.tensor([0.063, 0.123, 0.181, 0.237, 0.289, 0.337, 0.381, 0.422, 0.458, 0.490, 0.517, 0.540, 0.558, 0.571, 0.580, 0.584,
                                                                    0.583, 0.579, 0.570, 0.557, 0.540, 0.519, 0.495, 0.468, 0.438, 0.405, 0.370, 0.333, 0.294, 0.254, 0.213, 0.171, 0.129,
                                                                    0.087, 0.046, 0.005, -0.035, -0.074, -0.111, -0.146, -0.179, -0.210, -0.239, -0.265, -0.288, -0.309, -0.326, -0.341, -0.352, -0.360,
                                                                    -0.366, -0.368, -0.368, -0.364, -0.358, -0.350, -0.338, -0.325, -0.309, -0.291, -0.272, -0.250, -0.227, -0.204, -0.179, -0.153,
                                                                    -0.126,
                                                                    -0.100, -0.073, -0.046, -0.020, 0.006, 0.031, 0.055, 0.079, 0.101, 0.121, 0.141, 0.158, 0.174, 0.188, 0.201, 0.211, 0.220,
                                                                    0.226, 0.231, 0.233, 0.234, 0.233, 0.230, 0.225, 0.219, 0.211, 0.202, 0.191, 0.179, 0.166, 0.152],
                                                                   device=device))

    def fit_forward(self):
        self.fwd_solver.train_newton()

        us, xs = self.u_grid.get_real_u_x()
        return xs, us

    def fit_adjoint(self):
        """
        Fitting adjoint requires: dfdu, dfdu_x, dfdu_xx and L_u
        :return:
        """
        # Calculate dfdu, dfdu_x, dfdu_xx, dfdthetas
        dfdU, dfdtheta = self.pde_fwd.get_dfs()

        # Calculate L_u
        us = self.u_grid.get_with_bc()
        us_loss = us[2:-2]  # Shape = [N-2]
        L_u = self.loss_fn.quad_loss(us_loss)

        # Train adjoint
        self.adj_solver.train_newton((L_u, dfdU))

        self.dfdtheta = dfdtheta[1:-1]  # shape = [N-2, N_params]

    def backward(self):
        # Integrate adjoint to get gradeints
        a = self.a_grid.get_with_bc()[1:-1]  # Shape = [N-2]

        # dL/dtheta = integral[a * dfdtheta] dx
        dL = a.view(-1, 1) * self.dfdtheta  # Shape = [N-2, N_params]
        dLdtheta = torch.trapezoid(dL, self.u_grid.xs[2:-2], dim=0)

        # Set gradients. Instead of calling loss.backward(), we set the gradients manually
        self.pde_fwd.thetas.grad = dLdtheta
        return dLdtheta

    def plot(self):
        self.adj_solver.plot("Adjoint")
        self.fwd_solver.plot("Us")

    def parameters(self):
        return [self.pde_fwd.thetas]


def main():
    trainer = Trainer(device='cuda')
    optim = torch.optim.Adam(trainer.parameters(), lr=0.05)

    for i in range(100):
        print(i)
        us, xs = trainer.fit_forward()
        trainer.fit_adjoint()
        dLdtheta = trainer.backward()

        print(f'{dLdtheta = }')
        print("thetas = ", trainer.pde_fwd.thetas.cpu())

        optim.step()
        optim.zero_grad()

        print(us.shape, us.cpu())
        # exit(2)

    trainer.plot()


if __name__ == "__main__":
    main()
