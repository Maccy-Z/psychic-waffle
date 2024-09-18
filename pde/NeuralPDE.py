import torch
from cprint import c_print
from codetiming import Timer
import logging

from U_grid import UGridOpen2D
from pde.solvers.jacobian import get_jac_calc
from pdes.discrete_derivative import DerivativeCalc2D
from pdes.PDEs import PDEFunc
from pdes.PDE_utils import PDEForward, PDEAdjoint_tmp
from solvers.linear_solvers import SolverNewton, LinearSolver
from config import Config
from pde.loss import MSELoss, Loss

torch.set_printoptions(linewidth=150, precision=3)


class NeuralPDE:
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
        fwd_lin_solver = LinearSolver("iterative", cfg.DEVICE, cfg=fwd_cfg.lin_solve_cfg)
        fwd_jacob_calc = get_jac_calc(us_grid, pde_forward, fwd_cfg)
        newton_solver = SolverNewton(pde_forward, us_grid, fwd_lin_solver, jac_calc=fwd_jacob_calc, cfg=fwd_cfg)

        # Adjoint solver
        adj_lin_solver = LinearSolver("iterative", self.DEVICE, adj_cfg.lin_solve_cfg)
        adj_jacob_calc = get_jac_calc(us_grid, pde_forward, adj_cfg)
        pde_adjoint = PDEAdjoint_tmp(us_grid, pde_fn, deriv_calc, adj_jacob_calc, adj_lin_solver, loss_fn)

        self.pde_fn = pde_fn
        self.us_grid = us_grid
        self.newton_solver = newton_solver
        self.pde_adjoint = pde_adjoint

    def forward_solve(self):
        """ Solve PDE forward problem. """

        self.newton_solver.find_pde_root()
        us, _ = self.us_grid.get_real_us_Xs()
        # show_grid(us, "Fitted values")

    def adjoint_solve(self):
        """ Solve for adjoint """

        adjoint, loss = self.pde_adjoint.adjoint_solve()
        self.adjoint = adjoint

        c_print(f'loss: {loss.item():.4g}', "bright_magenta")
        return loss

    def backward(self):
        """ Once adjoint is calculated, backpropagate through PDE to get gradients.
            dL/dP = - adjoint * df/dP
         """
        residuals = self.pde_adjoint.backpropagate(self.adjoint)  # Shape = [N, ..., Nparams]

        # Delete adjoint to stop reuse.
        self.adjoint = None

        return residuals

    def get_us_Xs(self):
        us, Xs = self.us_grid.get_real_us_Xs()
        return us, Xs


def main():
    from pdes.PDEs import Poisson

    cfg = Config()
    pde_fn = Poisson(cfg, device=cfg.DEVICE)
    # pde_fn.plot_bc()

    N_us_grad = torch.isnan(pde_fn.dirichlet_bc).sum().item()   # Hardcoded
    us_base = torch.full((N_us_grad,), 0., device=cfg.DEVICE)
    loss_fn = MSELoss(us_base)

    pde_adj = NeuralPDE(pde_fn, loss_fn, cfg)

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


