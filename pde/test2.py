import torch
from cprint import c_print

from pde.NeuralPDE import NeuralPDE
from pdes.PDEs import Poisson, LearnedFunc
from config import Config
from pde.loss import MSELoss
from utils import show_grid, setup_logging

setup_logging()
def main():

    cfg = Config()
    pde_fn = LearnedFunc(cfg, device=cfg.DEVICE)

    optim = torch.optim.SGD(pde_fn.parameters(), momentum=0., lr=1.)

    us_base = torch.load('us.pt', weights_only=True) # torch.full((N_us_grad,), 0., device=cfg.DEVICE)
    loss_fn = MSELoss(us_base)

    pde_adj = NeuralPDE(pde_fn, loss_fn, cfg)

    for i in range(10):
        print()
        c_print(f'Iteration {i}', color="bright_cyan")
        pde_adj.forward_solve()
        pde_adj.adjoint_solve()
        pde_adj.backward()

        for n, p in pde_fn.named_parameters():
            print(f'p = {p.data.cpu()}')
            print(f'grad = {p.grad.data.cpu()}')

        optim.step()
        optim.zero_grad()

    us, Xs = pde_adj.get_us_Xs()
    show_grid(us, "us")


if __name__ == "__main__":
    main()

