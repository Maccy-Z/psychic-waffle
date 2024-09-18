import torch
from cprint import c_print

from X_grid import XGrid2D
from U_grid import UGridOpen2D
from pde.solvers.jacobian import get_jac_calc
from utils import show_grid
from pdes.discrete_derivative import DerivativeCalc2D
from pdes.PDEs import PDEFunc
from pdes.PDE_utils import PDEForward
from solvers.linear_solvers import SolverNewton, LinearSolver
from config import Config
from pde.loss import MSELoss, Loss

torch.set_printoptions(linewidth=150, precision=3)


class PDEAdjoint:
    us_grid: UGridOpen2D
    loss_fn: Loss
    adjoint: torch.Tensor

    def __init__(self, pde_fn: PDEFunc, loss_fn: Loss, cfg: Config):
        adj_cfg = cfg.adj_cfg
        fwd_cfg = cfg.fwd_cfg
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.DEVICE = cfg.DEVICE

        # Grid and BC
        Xs_grid = pde_fn.Xs_grid
        dirichlet_bc = pde_fn.dirichlet_bc
        neuman_bc = pde_fn.neuman_bc

        # PDE classes
        us_grid = UGridOpen2D(Xs_grid, dirichlet_bc=dirichlet_bc, neuman_bc=neuman_bc)
        deriv_calc = DerivativeCalc2D(Xs_grid, order=2)
        pde_forward = PDEForward(us_grid, pde_fn, deriv_calc)

        # Forward solver
        lin_newton = LinearSolver("iterative", cfg.DEVICE, cfg=fwd_cfg.lin_solve_cfg)
        fwd_jacob_calc = get_jac_calc(us_grid, pde_forward, fwd_cfg)
        solver = SolverNewton(pde_forward, us_grid, lin_newton, jac_calc=fwd_jacob_calc, cfg=fwd_cfg)

        # Adjoint solver
        adj_jacob_calc = get_jac_calc(us_grid, pde_forward, adj_cfg)
        adjoint_solver = LinearSolver("iterative", self.DEVICE, adj_cfg.lin_solve_cfg)

        self.pde_fn = pde_fn
        self.us_grid = us_grid
        self.pde_forward = pde_forward
        self.fwd_solver = solver
        self.adj_jacob_calc = adj_jacob_calc
        self.adjoint_solver = adjoint_solver

    def forward_solve(self):
        """ Solve PDE forward problem. """

        self.fwd_solver.find_pde_root()
        us, _ = self.us_grid.get_real_us_Xs()
        show_grid(us, "Fitted values")

    def adjoint_solve(self):
        """ Solve for adjoint """

        # Adjoint solves for lambda dgdu = J^T * lambda.
        with torch.no_grad():
            jacobian, _ = self.adj_jacob_calc.jacobian()
            jac_T = jacobian.T

        # One adjoint value for each trained u value, including boundary points.
        us, grad_mask, pde_mask = self.us_grid.get_us_mask()
        pde_mask_ = pde_mask[1:-1, 1:-1]
        us_grad = us[grad_mask]
        G = self.loss_fn(us_grad)
        G_u = self.loss_fn.gradient()

        adjoint = self.adjoint_solver.solve(jac_T, G_u)
        adj_view = torch.full(pde_mask_.shape, 0., device=self.DEVICE)  # Shape = [N]
        adj_view[pde_mask_] = adjoint
        # show_grid(adj_view, "adjoint")

        self.adjoint = adj_view

        loss = G.mean()
        print()
        c_print(f'loss: {loss.item():.4g}', "bright_magenta")
        return loss

    def backward(self):
        """ Once adjoint is calculated, backpropagate through PDE to get gradients.
            dL/dP = - adjoint * df/dP
         """

        dfdps = self.pde_forward.get_dfs()  # Shape = [N, ..., Nparams]

        dLdps = {}
        for n, dfdp in dfdps.items():
            dLdp = dfdp * self.adjoint[..., None]
            dLdp = - torch.sum(dLdp, dim=(0, 1))
            dLdps[n] = dLdp

        for n, p in self.pde_fn.named_parameters():
            p.grad = dLdps[n]

        # Delete adjoint
        self.adjoint = None


def main():
    from pdes.PDEs import Poisson

    cfg = Config()
    pde_fn = Poisson(cfg, device=cfg.DEVICE)
    # pde_fn.plot_bc()

    N_us_grad = torch.isnan(pde_fn.dirichlet_bc).sum().item()   # Hardcoded
    us_base = torch.full((N_us_grad,), 0., device=cfg.DEVICE)
    loss_fn = MSELoss(us_base)

    pde_adj = PDEAdjoint(pde_fn, loss_fn, cfg)

    pde_adj.forward_solve()
    pde_adj.adjoint_solve()
    pde_adj.backward()

    us, grad_mask, _ = pde_adj.us_grid.get_us_mask()
    us = us[grad_mask]
    torch.save(us, 'us.pt')

    for n, p in pde_fn.named_parameters():
        print(f'{n = }, {p.grad = }')

if __name__ == "__main__":
    main()


