import torch
import time
from matplotlib import pyplot as plt

from X_grid import XGrid2D
from U_grid import UGridOpen2D
from pde.solvers.jacobian import JacobCalc, SplitJacobCalc
from utils import show_grid
from pdes.discrete_derivative import DerivativeCalc2D
from pdes.PDEs import Poisson
from pdes.PDE_utils import PDEForward
from solvers.linear_solvers import SolverNewton, LinearSolver

torch.set_printoptions(linewidth=150, precision=3)

DEVICE = "cuda"

# 1) Init grid
xmin, xmax = 0, 1
N = torch.tensor([150, 150])
Xs_grid = XGrid2D(xmin, xmax, N, device=DEVICE)
grid_N = Xs_grid.N.tolist()

# 2) Dirichlet BC
dirichlet_bc = torch.full(grid_N, float('nan'))
dirichlet_bc[-1, :] = 1
# dirichlet_bc[0, :] = 0
dirichlet_bc[:, 0] = torch.arange(0, grid_N[0]) / (grid_N[0]-1)
dirichlet_bc[:, -1] = torch.arange(0, grid_N[0]) / (grid_N[0]-1)

# 3) Neuman BC
neuman_bc = torch.full(grid_N, float('nan'))
neuman_bc[0, 1:-1] = 1.
# neuman_bc[-1, 1:-1] = 1.
#
# plt.imshow(neuman_bc.T)
# plt.show()
#

# Init PDE classes
us_grid = UGridOpen2D(Xs_grid, dirichlet_bc=dirichlet_bc, neuman_bc=neuman_bc)
deriv_calc = DerivativeCalc2D(Xs_grid, order=2)
pde_fn = Poisson(device=DEVICE)
pde_handle = PDEForward(us_grid, pde_fn, deriv_calc)
lin_newton = LinearSolver("iterative", DEVICE, cfg={"maxiter": 1000, "restart": 125, "rtol": 1e-3})
solver = SolverNewton(pde_handle, us_grid, lin_newton, jac_mode="split", N_iter=3, lr=1)

# show_grid(us_grid.pde_mask, "PDE mask")

# Solve
solver.find_pde_root()

us, _ = us_grid.get_real_us_Xs()
show_grid(us, "Fitted values")
print()

# for n, p in dfdtheta.items():
#     print(f"{n}: {p.shape}")

# Get dfdu
jacob_calc = SplitJacobCalc(us_grid, pde_handle, num_blocks=4)
jacobian, _ = jacob_calc.jacobian()

# Adjoint solves for lambda dgdu = J^T * lambda.
# One adjoint value for each trained u value, including boundary points.
us, grad_mask, pde_mask = us_grid.get_us_mask()
pde_mask_ = pde_mask[1:-1, 1:-1]
us_grad = us[grad_mask]
g = us_grad ** 2
g_u = 2 * us_grad / g.numel() # dgdu hardcoded.
jac_T = jacobian.T

adj_cfg = {}
adjoint_solver = LinearSolver("iterative", DEVICE, adj_cfg)
adjoint = adjoint_solver.solve(jac_T, g_u)
adj_view = torch.full(pde_mask_.shape, 0., device=DEVICE)     # Shape = [N]
adj_view[pde_mask_] = adjoint
show_grid(adj_view * g.numel(), "adjoint")

# Get dfdtheta
dfdtheta = pde_handle.get_dfs()     # Shape = [N, ..., Nparams]

loss = g.mean()
print(f'{loss = }')
# Calculate loss gradeints
for n, p in dfdtheta.items():
    print(f"{n}: {p.shape}")

    dLdp = p * adj_view[..., None]

    dLdp = torch.sum(dLdp, dim=(0, 1))
    print(f'{dLdp = }')


#     print(p)
#     print()
# torch.save(dfdu, "dfdu.pth")
# jacobian = torch.zeros_like(dfdu[0])
# print(dfdu[0].shape)
# print(Xs_grid.dx)



# residuals = pde_handle.only_resid()
# print(residuals)
