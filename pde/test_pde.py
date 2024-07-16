import torch
import time

from X_grid import XGrid2D
from U_grid import UGridOpen2D
from utils import show_grid
from discrete_derivative import DerivativeCalc2D
from PDEs import Poisson
from PDE_utils import PDEForward
from pde_solve import SolverNewtonSplit, SolverNewton

torch.set_printoptions(linewidth=150, precision=3)

DEVICE = "cuda"

# Init grid
xmin, xmax = 0, 1
N = torch.tensor([151, 151])
Xs_grid = XGrid2D(xmin, xmax, N, device=DEVICE)
grid_N = Xs_grid.N.tolist()

# Dirichlet BC
dirichlet_bc = torch.full(grid_N, float('nan'))
# dirichlet_bc[-1, :] = 0
# dirichlet_bc[0, :] = 1  # torch.arange(1, grid_N[0]+1)
dirichlet_bc[:, 0] = 0
dirichlet_bc[:, -1] = 1

# Neuman BC
neuman_bc = torch.full(grid_N, float('nan'))
neuman_bc[0, 1:-1] = 0
neuman_bc[-1, 1:-1] = 0

print(f'{neuman_bc.shape = }')
print()

# Init PDE classes
us_grid = UGridOpen2D(Xs_grid, dirichlet_bc=dirichlet_bc, neuman_bc=neuman_bc)
deriv_calc = DerivativeCalc2D(Xs_grid, order=2)
pde_fn = Poisson()
pde_handle = PDEForward(us_grid, pde_fn, deriv_calc)
solver = SolverNewtonSplit(pde_handle, us_grid, solver="sparse", N_iter=50, lr=1)

# Solve
solver.find_pde_root()

us, _ = us_grid.get_real_us_Xs()
show_grid(us, "Fitted values")

# residuals = pde_handle.only_resid()
# print(residuals)
