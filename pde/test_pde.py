import torch

from X_grid import XGrid1D, XGrid2D
from U_grid import UGridOpen1D
from PDE_functions import PDE_forward
from pde_solve import PDESolver
from utils import show_grid

xmin, xmax = torch.tensor([0, 0]), torch.tensor([1, 1])

Xs_grid = XGrid2D(xmin, xmax, 0.2)
print(Xs_grid)
show_grid(Xs_grid.Xs[..., 1])
exit(7)

us_grid = UGridOpen1D(Xs_grid, dirichlet_bc={'x0_lower': 0}, neuman_bc={'x0_lower': 1})

pde_fn = PDE_forward(us_grid)

solver = PDESolver(pde_fn, us_grid, N_iter=40, lr=2)

solver.train_newton()

u, x = us_grid.get_real_u_x()

show_grid(u)
print(Xs_grid)
