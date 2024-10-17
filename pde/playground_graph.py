import torch
from matplotlib import pyplot as plt

from pde.graph_grid.graph_store import Point, P_Types
from pde.graph_grid.U_graph import UGraph
from pde.graph_grid.graph_utils import test_graph, gen_perim, plot_interp_graph
from pde.config import Config
from pde.NeuralPDE_Graph import NeuralPDEGraph
from pdes.PDEs import Poisson, LearnedFunc
from pde.utils import setup_logging
from pde.loss import DummyLoss

def main():
    torch.set_printoptions(linewidth=200, precision=3)
    setup_logging()
    cfg = Config()

    # Make computation graph
    Xs_perim = gen_perim(1, 1, 0.1)
    Xs_bulk = test_graph(0.02, 0.98, torch.tensor([12, 12]), device="cpu")
    Xs_bc = [Point(P_Types.BOUNDARY, X, 0.) for X in Xs_perim]
    Xs_bulk = [Point(P_Types.NORMAL, X, 0.) for X in Xs_bulk]
    Xs_all = {i: X for i, X in enumerate(Xs_bc + Xs_bulk)}
    u_graph = UGraph(Xs_all, device=cfg.DEVICE)


    pde_fn = LearnedFunc(cfg, device=cfg.DEVICE)
    optim = torch.optim.Adam(pde_fn.parameters(), lr=0.5)

    pde_adj = NeuralPDEGraph(pde_fn, u_graph, cfg, DummyLoss())

    for i in range(5):
        pde_adj.forward_solve()

        loss = pde_adj.adjoint_solve()
        pde_adj.backward()

        print(f'{loss = :.3g}')
        for n, p in pde_fn.named_parameters():
            print(f'p = {p.data.cpu()}')
            print(f'grad = {p.grad.data.cpu()}')

        optim.step()
        optim.zero_grad()

    us, Xs = u_graph.us, u_graph.Xs

    plot_interp_graph(Xs, us)


if __name__ == "__main__":
    main()


