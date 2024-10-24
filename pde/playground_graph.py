import torch

from pde.graph_grid.graph_store import Point, P_Types
from pde.graph_grid.U_graph import UGraph
from pde.graph_grid.graph_utils import test_grid, gen_perim, plot_interp_graph
from pde.config import Config
from pde.NeuralPDE_Graph import NeuralPDEGraph
from pdes.PDEs import Poisson, LearnedFunc
from pde.utils import setup_logging
from pde.loss import DummyLoss


def new_graph(cfg):
    cfg = Config()

    Xs_perim = gen_perim(1, 1, 0.1)
    perim_mask = (Xs_perim[:, 0] == 0)
    Xs_neumann = Xs_perim[perim_mask]
    Xs_dirich = Xs_perim[~perim_mask]


    Xs_bulk = test_grid(0.02, 0.98, torch.tensor([10, 10]), device="cpu")
    #Xs_bc = [Point(P_Types.FIX_BC, X, value=0.) for X in Xs_perim]
    Xs_fix = [Point(P_Types.DirichBC, X, value=0.) for X in Xs_dirich]
    deriv = ([(1, 0)], 0.)
    Xs_deriv = [Point(P_Types.NeumOffsetBC, X, value=0., derivatives=deriv) for X in Xs_neumann]

    Xs_bulk = [Point(P_Types.Normal, X, value= 0.) for X in Xs_bulk]
    Xs_all = {i: X for i, X in enumerate(Xs_deriv + Xs_fix + Xs_bulk)}
    u_graph = UGraph(Xs_all, grad_acc=4, device=cfg.DEVICE)

    with open("save_u_graph.pth", "wb") as f:
        torch.save(u_graph, f)

    return u_graph

def load_graph(cfg):
    u_graph = torch.load("save_u_graph.pth")
    return u_graph

def true_pde():
    cfg = Config()
    u_graph = new_graph(cfg)

    pde_fn = Poisson(cfg, device=cfg.DEVICE)
    pde_adj = NeuralPDEGraph(pde_fn, u_graph, cfg, DummyLoss())

    pde_adj.forward_solve()
    us, Xs = u_graph.us, u_graph.Xs
    plot_interp_graph(Xs, us)

def main():
    torch.set_printoptions(linewidth=200, precision=3)
    cfg = Config()

    # Make computation graph
    Xs_perim = gen_perim(1, 1, 0.1)
    Xs_bulk = test_grid(0.02, 0.98, torch.tensor([12, 12]), device="cpu")
    Xs_bc = [Point(P_Types.FIX, X, 0.) for X in Xs_perim]
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
    setup_logging()
    torch.manual_seed(1)

    true_pde()


