import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
import abc
import math


class DerivativeCalculator(abc.ABC):
    def __init__(self, dx, order, device='cpu'):
        self.device = device
        if order == 2:
            self.du_kern_raw = [-0.5, 0, 0.5]
            self.d2u_ker_raw = [1, -2, 1]
        elif order == 4:
            self.du_kern_raw = [1 / 12, -2 / 3, 0, 2 / 3, -1 / 12]
            self.d2u_ker_raw = [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12]
        else:
            raise ValueError(f"Given order {order} not supported. Must be 2, 4, 6, 8")

        # Boundary coefficients
        self.du_forward = [-1.5, 2, -0.5]
        self.du_backward = [0.5, -2, 1.5]
        self.d2u_forward = [2, -5, 4, -1]
        self.d2u_backward = [-1, 4, -5, 2]

    @abc.abstractmethod
    def derivative(self, u):
        pass


class DerivativeCalc1D(DerivativeCalculator):
    def __init__(self, dx, order, **kwargs):
        super().__init__(dx, order, **kwargs)

        # Central kernels
        self.du_kernel = torch.tensor([self.du_kern_raw], dtype=torch.float32).reshape(1, 1, -1).to(self.device) / dx
        self.d2u_kernel = torch.tensor([self.d2u_ker_raw], dtype=torch.float32).reshape(1, 1, -1).to(self.device) / (dx ** 2)

        # Edge forward/backward kernels
        self.du_forward_kernel = torch.tensor([self.du_forward], dtype=torch.float32).reshape(1, 1, -1).to(self.device) / dx
        self.du_backward_kernel = torch.tensor([self.du_backward], dtype=torch.float32).reshape(1, 1, -1).to(self.device) / dx
        self.d2u_forward_kernel = torch.tensor([self.d2u_forward], dtype=torch.float32).reshape(1, 1, -1).to(self.device) / (dx ** 2)
        self.d2u_backward_kernel = torch.tensor([self.d2u_backward], dtype=torch.float32).reshape(1, 1, -1).to(self.device) / (dx ** 2)

    def derivative(self, u):
        """ u.shape = [N + boundary]
            Return shape: [N]
        """
        u = u.view(1, *u.shape)
        dudx = F.conv1d(u, self.du_kernel, padding=0, stride=1)[0]  # [0, self.extra_points:-self.extra_points]
        d2udx2 = F.conv1d(u, self.d2u_kernel, padding=0, stride=1)[0]

        return dudx, d2udx2

    def derivative_boundary(self, u):
        """
        Derivative, including boundary points. Approxed using forward/backward difference.
            u.shape = [N]
            Return shape: [N]
        """
        dudx, d2udx2 = torch.empty_like(u), torch.empty_like(u)
        # Main points
        dudx_cent, d2udx2_cent = self.derivative(u)
        dudx[1:-1] = dudx_cent
        d2udx2[1:-1] = d2udx2_cent

        # Boundary points
        du_low, d2u_low = self._forward_diff(u)
        du_high, d2u_high = self._backward_diff(u)
        dudx[0] = du_low
        dudx[-1] = du_high
        d2udx2[0] = d2u_low
        d2udx2[-1] = d2u_high

        return dudx, d2udx2

    def _forward_diff(self, u):
        """Forward derivative for boundary points"""

        u = u.view(1, *u.shape)
        dudx_low = F.conv1d(u[:, :3], self.du_forward_kernel, padding=0, stride=1)[0]
        d2udx2_low = F.conv1d(u[:, :4], self.d2u_forward_kernel, padding=0, stride=1)[0]

        return dudx_low, d2udx2_low

    def _backward_diff(self, u):
        """Backward derivative for boundary points"""

        u = u.view(1, *u.shape)
        dudx_high = F.conv1d(u[:, -3:], self.du_backward_kernel, padding=0, stride=1)[0]
        d2udx2_high = F.conv1d(u[:, -4:], self.d2u_backward_kernel, padding=0, stride=1)[0]

        return dudx_high, d2udx2_high


class DerivativeCalc2D(DerivativeCalculator):
    def __init__(self, dx, order, extra_points):
        super().__init__(dx, order, extra_points)

        # Kernel shape: [out_channels, in_channels, kern_x, kern_y]
        self.dudx_kern = torch.tensor([self.du_kern_raw], dtype=torch.float32).reshape(1, 1, -1, 1) / dx
        self.dudy_kern = torch.tensor([self.du_kern_raw], dtype=torch.float32).reshape(1, 1, 1, -1) / dx

        self.d2udx_kern = torch.tensor([self.d2u_ker_raw], dtype=torch.float32).reshape(1, 1, -1, 1) / (dx ** 2)
        self.d2udy_kern = torch.tensor([self.d2u_ker_raw], dtype=torch.float32).reshape(1, 1, 1, -1) / (dx ** 2)

    def derivative(self, u) -> (torch.Tensor, torch.Tensor):
        """ u.shape = [Nx + boundary, Ny + boundary]
            Return shape: [2, Nx, Ny]
        """
        u = u.view(1, *u.shape)  # Shape: [BS, channels, Nx, Ny]

        dudx = F.conv2d(u, self.dudx_kern, padding=0, stride=1)[0, :, self.extra_points:-self.extra_points]
        dudy = F.conv2d(u, self.dudy_kern, padding=0, stride=1)[0, self.extra_points:-self.extra_points, :]

        d2udx2 = F.conv2d(u, self.d2udx_kern, padding=0, stride=1)[0, :, self.extra_points:-self.extra_points]
        d2udy2 = F.conv2d(u, self.d2udy_kern, padding=0, stride=1)[0, self.extra_points:-self.extra_points, :]

        dudX = torch.stack([dudx, dudy], dim=0)
        d2udX2 = torch.stack([d2udx2, d2udy2], dim=0)

        return dudX, d2udX2


class PointGrid:
    """ Grid of X points to be shared between classes"""

    def __init__(self, Xmin, Xmax, N, device='cpu'):
        self.device = device

        self.L = Xmax - Xmin
        self.N = N
        self.dx = self.L / (N - 1)
        self.xs = torch.linspace(Xmin - self.dx, Xmax + self.dx, N + 2).to(self.device)  # Shape: [N + 2], [Xmin-dx, Xmin, ...,Xmin, Xmin+dx]


class PDEGrid(abc.ABC):
    """
    Contains grid of points for solving PDE.
    Handles and saves boundary conditions and masks.
    """
    us: torch.Tensor
    pre_approved_bc: set = {'x0_lower', 'x0_upper'}
    neuman_bc: dict
    dirichlet_bc: dict
    grad_mask: torch.Tensor
    u_mask: torch.Tensor

    def __init__(self, X_grid: PointGrid, dirichlet_bc: dict = None, neuman_bc: dict = None, device='cpu'):
        self.device = device
        self._clean_dict(dirichlet_bc, neuman_bc)

        self.dx = X_grid.dx

    def _clean_dict(self, dirichlet_bc, neuman_bc):
        if dirichlet_bc is None:
            dirichlet_bc = dict()
        if neuman_bc is None:
            neuman_bc = dict()
        assert set(dirichlet_bc.keys()).issubset(self.pre_approved_bc), f'{dirichlet_bc.keys()} not in {self.pre_approved_bc}'
        assert set(neuman_bc.keys()).issubset(self.pre_approved_bc), f'{neuman_bc.keys()} not in {self.pre_approved_bc}'
        self.dirichlet_bc = dirichlet_bc
        self.neuman_bc = neuman_bc

    def remove_bc(self, us):
        """
        Given an array, remove points that don't have gradient.
        """
        us = us[self.u_mask]
        return us

    def update_grid(self, deltas):
        """
        deltas.shape = [N]
        us -> us - deltas
        """
        self.us[self.grad_mask] -= deltas

    def set_grid(self, new_us):
        self.us[self.grad_mask] = new_us

    def get_real(self):
        return self.us[self.u_mask], self.xs[self.u_mask]

    def get_with_bc(self):
        """
        Return grid of points with boundary conditions set.
        """
        pass


class PDEGridOpen(PDEGrid):
    def __init__(self, X_grid: PointGrid,
                 dirichlet_bc: dict = None, neuman_bc: dict = None,
                 device='cpu'):
        """
        :param L: Size of domain
        :param N: Number of points PDE is enforced at
        :param dirichlet_bc: Dict of {'left'/'right': Value at bc}. Set inside the grid
        :param neuman_bc: Dict of {'left'/'right': Value at bc}. Set outside the grid

        Add on extra terms at end of grid so derivatives can be taken at the boundary if needed.
        PDE is constrained at N middle points (u0 - un-1), but derivatives are taken at N - N_boundary points.
        Format of grid:
            Only dirichlet:                 [E, u0=DL, u1,..., un-1=DR, E]
            Dirichlet and Neuman on left:   [NL, u0=DL, u1,..., un-1, E]
            Dirichlet left, Neuman right:   [E, u0=DL, u1,..., un-1, NR]
            Neuman only:                    [NL, u0, u1,..., un-1, NR]
        """
        super().__init__(X_grid, dirichlet_bc, neuman_bc, device)

        # Grid of points.
        self.xs = X_grid.xs
        self.us = torch.ones_like(self.xs).to(self.device)  # Shape: [N + 2]

        # Mask of where to compute gradients, no gradient at boundary
        self.grad_mask = torch.ones_like(self.xs).to(torch.bool).to(self.device)
        if 'x0_lower' in self.dirichlet_bc:
            self.grad_mask[1] = 0
        if 'x0_upper' in self.dirichlet_bc:
            self.grad_mask[-2] = 0
        if 'x0_lower' in self.neuman_bc:
            self.grad_mask[0] = 0
        if 'x0_upper' in self.neuman_bc:
            self.grad_mask[-1] = 0

        # Mask of which elements are real
        self.u_mask = torch.ones_like(self.xs).to(torch.bool).to(self.device)
        self.u_mask[0] = 0
        self.u_mask[-1] = 0

    def get_with_bc(self):
        if 'x0_lower' in self.dirichlet_bc:
            self.us[1] = self.dirichlet_bc['x0_lower']
        if 'x0_upper' in self.dirichlet_bc:
            self.us[-2] = self.dirichlet_bc['x0_upper']

        # Compute imaginary value so derivatives are as given.
        if 'x0_lower' in self.neuman_bc:
            self.us[0] = self.us[2] - 2 * self.neuman_bc['x0_lower'] * self.dx
        if 'x0_upper' in self.neuman_bc:
            self.us[-1] = self.us[-3] + 2 * self.neuman_bc['x0_upper'] * self.dx
        return self.us


class PDEGridClosed(PDEGrid):
    """
    Like a PDEGrid, but setting neuman boundary conditions with interior points instead of fantasy.
    """

    def __init__(self, X_grid: PointGrid, dirichlet_bc: dict = None, neuman_bc: dict = None, device='cpu'):
        """
        :param L: Size of domain
        :param N: Number of points PDE is enforced at
        :param dirichlet_bc: Dict of {'left'/'right': Value at bc}. Set inside the grid
        :param neuman_bc: Dict of {'left'/'right': Value at bc}. Set outside the grid

        Add on extra terms at end of grid so derivatives can be taken at the boundary if needed.
        PDE is constrained at N middle points (u0 - un-1), but derivatives are taken at N - N_boundary points.
        Format of grid:
            Only dirichlet:                 [E, u0=DL, u1,..., un-1=DR, E]
            Dirichlet and Neuman on left:   [NL, u0=DL, u1,..., un-1, E]
            Dirichlet left, Neuman right:   [E, u0=DL, u1,..., un-1, NR]
            Neuman only:                    [NL, u0, u1,..., un-1, NR]
        """
        super().__init__(X_grid, dirichlet_bc, neuman_bc, device)

        # Grid of points.
        self.xs = X_grid.xs[1:-1]
        self.us = torch.ones_like(self.xs).to(self.device)  # Shape: [N]

        # Mask of where to compute gradients, no gradient at boundary
        self.grad_mask = torch.ones_like(self.xs).to(torch.bool).to(self.device)
        if 'x0_lower' in self.dirichlet_bc:
            self.grad_mask[0] = 0
        if 'x0_upper' in self.dirichlet_bc:
            self.grad_mask[-1] = 0
        if 'x0_lower' in self.neuman_bc:  # Don't use imaginary boundary if not needed
            self.grad_mask[1] = 0
        if 'x0_upper' in self.neuman_bc:
            self.grad_mask[-2] = 0

        # Mask of which elements are real
        self.u_mask = torch.ones_like(self.xs).to(torch.bool).to(self.device)

    def get_with_bc(self):
        if 'x0_lower' in self.dirichlet_bc:
            self.us[0] = self.dirichlet_bc['x0_lower']
        if 'x0_upper' in self.dirichlet_bc:
            self.us[-1] = self.dirichlet_bc['x0_upper']

        # Use second order gradient approximation on boundary.
        if 'x0_lower' in self.neuman_bc:
            self.us[1] = 1 / 4 * (2 * self.neuman_bc['x0_lower'] * self.dx + 3 * self.us[0] + self.us[2])
        if 'x0_upper' in self.neuman_bc:
            self.us[-2] = 1 / 4 * (2 * self.neuman_bc['x0_upper'] * self.dx - 3 * self.us[-1] - self.us[-3])

        return self.us


def main1D():
    # Parameters
    Xmin, Xmax = 0, 1  # 2 * math.pi
    N = 100
    X_grid = PointGrid(Xmin, Xmax, N)

    u_grid = PDEGrid(X_grid, dirichlet_bc={'x0_lower': 0, 'x0_upper': 1})
    a_grid = PDEGridClosed(X_grid, dirichlet_bc={'x0_lower': 0, 'x0_upper': 0}, neuman_bc={'x0_lower': 0, 'x0_upper': 2})

    xs = X_grid.xs
    y = torch.sin(xs)
    print(a_grid.get_with_bc())
    print(a_grid.get_with_bc().shape)


def main2D():
    def f(Xs):
        # Xs.shape = [Nx, Ny, 2]
        x = Xs[..., 0]
        y = Xs[..., 1]
        return torch.sin(x) * torch.cos(y)

    # Parameters
    L = 2 * torch.pi  # Length of the spatial domain
    N = 5  # Number of points
    dx = L / N  # Resolution of discretisation
    order = 2  # Order of derivative
    extra_points = order // 2

    derivative_calculator = DerivativeCalc2D(dx, order=order, extra_points=extra_points)

    # Tensor of points, with additional boundary points
    x_idx = torch.arange(N + extra_points * 2)
    y_idx = torch.arange(N + extra_points * 2)
    x, y = torch.meshgrid(x_idx, y_idx, indexing='xy')
    Xs = torch.stack([x, y], dim=-1)
    Xs = (Xs - extra_points) * L / N
    us = f(Xs).to(torch.float32)

    dudX, d2udX2 = derivative_calculator.derivative(us)

    # Remove extra points for plotting
    us = us[extra_points:-extra_points, extra_points:-extra_points].numpy().squeeze()
    dudX = dudX.numpy()
    d2udX2 = d2udX2.numpy()

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(us, origin='lower')
    plt.title('Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.imshow(d2udX2[1], origin='lower', vmin=-1, vmax=1)
    # plt.imshow(d2udx2, label="y'' = -sin(x)")
    plt.title('Discrete Derivative')
    plt.xlabel('x')
    plt.ylabel("y")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main1D()
