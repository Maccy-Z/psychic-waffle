import torch
import abc


class XGrid(abc.ABC):
    """ Grid of X points to be shared between classes """
    N_dim: int
    Xs: torch.Tensor  # X points, xs.shape = [[N]* N_dim]
    dx: float  # Spacing between points

    def __init__(self, device='cpu'):
        """ Xmin: minimum of grid
            Xmax: maximum of grid
            N: number of points in grid
            device: device to store grid on
        """
        self.device = device

    def __repr__(self):
        return f"{self.N_dim}D X points (including extra boundary points): " + str(self.Xs)


class XGrid1D(XGrid):
    N_dim = 1

    def __init__(self, Xmin, Xmax, dx: float, device='cpu'):
        super().__init__(device)
        self.dx = dx
        self.L = Xmax - Xmin
        N = self.L / dx
        assert torch.all(torch.eq(N, torch.round(N))), f'{N = } is not an integer'

        self.Xs = torch.linspace(Xmin - self.dx, Xmax + self.dx, N + 2).to(self.device)  # Shape: [N + 2], [Xmin-dx, Xmin, ...,Xmin, Xmin+dx]


class XGrid2D(XGrid):
    N_dim = 2

    def __init__(self, Xmin: torch.Tensor, Xmax: torch.Tensor, dx: float, device='cpu'):
        super().__init__(device)
        self.dx = dx
        self.L = Xmax - Xmin
        N = self.L / dx
        assert torch.all(torch.eq(N, torch.round(N))), f'{N = } is not an integer'

        self.N = N.to(torch.int) + 1

        x_values = torch.arange(Xmin[0] - dx, Xmax[0] + dx, dx)
        y_values = torch.arange(Xmin[1] - dx, Xmax[1] + dx, dx)

        # Create the grid of points
        n_grid, m_grid = torch.meshgrid(x_values, y_values, indexing='xy')

        # Combine the grids to form the final N x M grid of points
        self.Xs = torch.stack([n_grid, m_grid], dim=-1).to(self.device)


def main():
    xmin, xmax = torch.tensor([0, 0]), torch.tensor([1, 1])
    Xs = XGrid2D(xmin, xmax, 0.1)


if __name__ == "__main__":
    main()
