import torch
import time

from X_grid import XGrid2D
from U_grid import UGridOpen2D
from utils import show_grid
from DiscreteDerivative import DerivativeCalc2D
from PDEs import Poisson
from PDE_utils import PDEForward
from pde_solve import SolverNewtonSplit, SolverNewton

torch.set_printoptions(linewidth=150, precision=3)

DEVICE = "cpu"

# Init grid
xmin, xmax = torch.tensor([0, 0]), torch.tensor([1, 1])
N = torch.tensor([11, 11])
Xs_grid = XGrid2D(xmin, xmax, N, device=DEVICE)
grid_N = Xs_grid.N.tolist()

# Init BC
dirichlet_bc = torch.full(grid_N, float('nan'))
# dirichlet_bc[-1, :] = 0
# dirichlet_bc[0, :] = 1  # torch.arange(1, grid_N[0]+1)

dirichlet_bc[:, 0] = 0
dirichlet_bc[:, -1] = 1

neuman_bc = torch.full(grid_N, float('nan'))
neuman_bc[0, 1:-1] = 2
neuman_bc[-1, 1:-1] = 0

print(f'{neuman_bc.shape = }')
print()

# Init PDE classes
us_grid = UGridOpen2D(Xs_grid, dirichlet_bc=dirichlet_bc, neuman_bc=neuman_bc)
deriv_calc = DerivativeCalc2D(Xs_grid, order=2)
pde_fn = Poisson()
pde_handle = PDEForward(us_grid, pde_fn, deriv_calc)
solver = SolverNewton(pde_handle, us_grid, N_iter=5, lr=1.)

# # Coordinates
# us_bc, Xs_bc = us_grid.get_all_us_Xs()
# us, Xs = us_grid.get_real_us_Xs()

# show_grid(us_grid.grad_mask, "grad mask")
# show_grid(us_grid.neuman_bc, title="Neuman BC")
# show_grid(us_grid.dirichlet_bc, title='Dirichlet BC')
# show_grid(us_grid.pde_mask, title='PDE mask')

# Solve
solver.find_pde_root()

us, _ = us_grid.get_real_us_Xs()
show_grid(us, "Fitted values")

# residuals = pde_handle.only_resid()
# show_grid(residuals, "Residuals")
