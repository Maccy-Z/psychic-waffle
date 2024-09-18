import torch

from PDEAdjoint import PDEAdjoint
from pdes.PDEs import Poisson, LearnedFunc
from config import Config
from pde.loss import MSELoss

def main():

    cfg = Config()
    pde_fn = LearnedFunc(cfg, device=cfg.DEVICE)

    us_base = torch.load('us.pt', weights_only=True) # torch.full((N_us_grad,), 0., device=cfg.DEVICE)
    loss_fn = MSELoss(us_base)

    pde_adj = PDEAdjoint(pde_fn, loss_fn, cfg)

    pde_adj.forward_solve()
    pde_adj.adjoint_solve()
    pde_adj.backward()

    us, Xs = pde_adj.us_grid.get_real_us_Xs()

    for n, p in pde_fn.named_parameters():
        print(f'{n = }, {p.grad = }')

if __name__ == "__main__":
    main()

