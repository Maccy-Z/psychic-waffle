import torch
from abc import ABC, abstractmethod


class PDEFunc(torch.nn.Module, ABC):
    def __init__(self, device='cpu'):
        """ Given u and derivatives, return the PDE residual"""
        super().__init__()

    def residuals(self, u_dus: tuple[torch.Tensor, ...], Xs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u_dus: u, du/dx, d2u/dx2. Shape = [3][Nitem, Nx, Ny].
            Xs: Grid points. Shape = [2, Nx, Ny]

        f(u, du/dX, d2u/dX2, X, thetas) = 0

        Returns: PDE residual (=0 for exact solution), shape=[Nx, Ny]
        """

        return self.forward(u_dus, Xs)



class Poisson(PDEFunc):
    def __init__(self, device='cpu'):
        super().__init__(device=device)

        self.test_param = torch.nn.Parameter(torch.tensor([1., 1.], device=device))
        self.to(device)


    def forward(self, u_dus: tuple[torch.Tensor, ...], Xs: torch.Tensor):
        u, dudX, d2udX2 = u_dus
        u = u[0]
        # print(f'{u.shape = }, {d2udX2.shape = }, {Xs.shape = }')

        x_min, x_max = 0.25, 0.5
        y_min, y_max = 0.5, 0.6

        x, y = Xs

        x_masks = (x > x_min) & (x < x_max)
        y_masks = (y > y_min) & (y < y_max)
        charge = 100 * (x_masks & y_masks)


        resid = d2udX2[0] + 0.5 * d2udX2[1] + 50 * u  + 50 * self.test_param[0]# +  0 * charge # + 1 * self.test_param[0] + 0.5 * self.test_param[1]
        # resid += 10 * self.test_param[0]

        #print(f'{resid.shape = }, ')
        return resid
