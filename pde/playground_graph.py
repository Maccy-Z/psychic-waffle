import torch
from matplotlib import pyplot as plt

from pde.graph_grid.graph_store import Point, P_Types
from pde.graph_grid.U_graph import UGraph
from pde.graph_grid.graph_utils import test_graph, gen_perim, plot_interp_graph
from pde.config import Config
from pde.NeuralPDE_Graph import NeuralPDEGraph
from pdes.PDEs import Poisson
from pde.utils import setup_logging

def main():
    torch.set_printoptions(linewidth=200, precision=3)
    setup_logging()
    cfg = Config()

    Xs_perim = gen_perim(1, 1, 0.1)
    Xs_bulk = test_graph(0.01, 0.99, torch.tensor([45, 45]), device="cpu")
    Xs_bc = [Point(P_Types.BOUNDARY, X, 0.) for X in Xs_perim]
    Xs_bulk = [Point(P_Types.NORMAL, X, 0.) for X in Xs_bulk]
    Xs_all = {i: X for i, X in enumerate(Xs_bc + Xs_bulk)}
    u_graph = UGraph(Xs_all, device=cfg.DEVICE)

    pde_fn = Poisson(cfg, device=cfg.DEVICE)

    pde_adj = NeuralPDEGraph(pde_fn, u_graph, cfg, None)
    pde_adj.forward_solve()

    us, Xs = u_graph.us, u_graph.Xs

    plot_interp_graph(Xs, us)



if __name__ == "__main__":
    main()


