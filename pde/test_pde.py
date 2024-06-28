import torch

from X_grid import XGrid1D, XGrid2D
from U_grid import UGridOpen1D, UGridOpen2D
from utils import show_grid
from DiscreteDerivative import DerivativeCalc2D
from PDEs import Poission
xmin, xmax = torch.tensor([0, 0]), torch.tensor([1, 1])

Xs_grid = XGrid2D(xmin, xmax, 0.2)

dirichlet_bc = torch.ones([6, 6])
dirichlet_bc[:] = float('nan')
dirichlet_bc[:, 0] = torch.arange(1, 7)

neuman_bc = torch.ones([6, 6])
neuman_bc[1:-1, 1:-1] = float('nan')
neuman_bc[1, -1] = float('nan')
neuman_bc[3, -1] = -1

us_grid = UGridOpen2D(Xs_grid, dirichlet_bc=dirichlet_bc, neuman_bc=neuman_bc)
us_bc, Xs_bc = us_grid.get_us_Xs()
us, Xs = us_grid.get_real_us_Xs()
deriv_calc = DerivativeCalc2D(Xs_grid, order=2)
du_dX, d2u_dX2 = deriv_calc.derivative(us_bc)

print(f'{us.shape = }, {Xs.shape = }, {du_dX.shape = }, {d2u_dX2.shape = }')
show_grid(us_bc)
show_grid(d2u_dX2[0])

pde_fn = Poission()
us_dus = [us, du_dX, d2u_dX2]
residuals = pde_fn.residuals(us_bc, None)

# us_grid = UGridOpen1D(Xs_grid, dirichlet_bc={'x0_lower': 0}, neuman_bc={'x0_lower': 1})
#
# pde_fn = PDE_forward(us_grid)
#
# solver = PDESolver(pde_fn, us_grid, N_iter=40, lr=2)
#
# solver.train_newton()
#
# u, x = us_grid.get_real_u_x()
#
# show_grid(u)
# print(Xs_grid)
