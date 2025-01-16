import torch
import math

from pde.graph_grid.graph_store import Point, P_Types, Deriv
from pde.graph_grid.U_graph import UGraph
from pde.graph_grid.graph_utils import test_grid, gen_perim, plot_interp_graph
from pde.config import Config
from pdes.PDEs import Poisson, LearnedFunc
from pde.utils import setup_logging
from pde.mesh_generation.subproc_gen_mesh import run_subprocess

def new_graph(cfg):
    cfg = Config()
    N_comp = 2

    n_grid = 20
    spacing = 1/(n_grid + 1)

    deriv = [Deriv(comp=[0], orders=[(1, 0)], value=1.), Deriv(comp=[1], orders=[(1, 0)], value=0.)]

    Xs_perim = gen_perim(1, 1, spacing)
    perim_mask = (Xs_perim[:, 1] > 0) & (Xs_perim[:, 1] < 1) & (Xs_perim[:, 0] ==0)
    Xs_neumann = Xs_perim[perim_mask]
    #print(Xs_neumann)
    Xs_dirich = Xs_perim[~perim_mask]

    Xs_ghost = Xs_neumann.clone()
    #Xs_ghost[:, 0] = Xs_ghost[:, 0] - spacing
    Xs_bulk = test_grid(spacing, (1- spacing), torch.tensor([n_grid, n_grid]), device="cpu")

    Xs_fix = [Point(P_Types.DirichBC, X, value=[0. for _ in range(N_comp)]) for X in Xs_dirich]
    Xs_deriv = [Point(P_Types.NeumOffsetBC , X, value=[0. for _ in range(N_comp)], derivatives=deriv) for X in Xs_neumann]
    Xs_ghost = [] # [Point(P_Types.Ghost, X, value=[0. for _ in range(N_comp)]) for X in Xs_ghost]
    Xs_bulk = [Point(P_Types.Normal, X, value= [0. for _ in range(N_comp)]) for X in Xs_bulk]


    Xs_all = {i: X for i, X in enumerate(Xs_deriv + Xs_fix + Xs_bulk + Xs_ghost)}
    edge_mask = []
    for i, point in Xs_all.items():
        X = point.X
        x, y = X[0].item(), X[1].item()
        point.value = [x * y**2]
        if x < 0.01:
            point.edge_mask=False
            #point.value = [0]
        else:
            point.edge_mask=True
            #point.value = [1]

    edge_mask = torch.tensor(edge_mask, dtype=torch.float32)
    u_graph = UGraph(edge_mask, Xs_all, N_component=N_comp, grad_acc=2, device=cfg.DEVICE)

    with open("save_u_graph.pth", "wb") as f:
        torch.save(u_graph, f)

    return u_graph

def load_graph(cfg):
    u_graph = torch.load("save_u_graph.pth")
    return u_graph

def true_pde():
    cfg = Config()
    #u_graph = load_graph(cfg)
    #u_graph = mesh_graph(cfg)
    u_graph = new_graph(cfg)

    us, Xs = u_graph.get_all_us_Xs()
    plot_interp_graph(Xs, us[:, 0], title="Initial")

    u_graph.set_eval_deriv_calc(torch.ones_like(u_graph.pde_mask))
    grads = u_graph.deriv_calc_eval.derivative(us)
    #torch.save((grads, us), "boundary_grads.pth")
    grads_bc, us_mask = torch.load("boundary_grads.pth")

    lims = [-0, 6]
    laplace_ideal = grads["laplacian"]
    plot_interp_graph(Xs, laplace_ideal[:, 0], lim=lims, title="Ideal Laplace")

    laplace_approx = grads[(2, 0)] + grads[(0, 2)]
    plot_interp_graph(Xs, laplace_approx[:, 0], lim=lims, title="Approx Laplace")

    true_laplace = 6 * (Xs[:, 1]).unsqueeze(1)
    plot_interp_graph(Xs, true_laplace[:, 0], lim=lims, title="True Laplace")

    g_u_d_g_f = grads_bc[(1, 0)] * grads[(1, 0)] + grads_bc[(0, 1)] * grads[(0, 1)]
    bc_laplace = g_u_d_g_f + us_mask * (grads[(2, 0)] + grads[(0, 2)])
    plot_interp_graph(Xs, bc_laplace[:, 0], lim=lims, title="BC Laplace")

    # ideal_diff = - true_laplace + laplace_ideal
    # approx_diff = true_laplace - laplace_approx
    # ideal_mae = torch.mean(ideal_diff.abs())
    # approx_mae = torch.mean(approx_diff.abs()) / ideal_mae
    # print(f'{ideal_mae * 100 = }, {approx_mae = }')
    # plot_interp_graph(Xs, ideal_diff[:, 0], title="Ideal diff")
    # plot_interp_graph(Xs, laplace_ideal[:, 0], title="Ideal Laplace")
    # plot_interp_graph(Xs, approx_diff[:, 0], title="Approx diff")
    # plot_interp_graph(Xs, combine_diff[:, 0], title="Combine diff")

    # # Gradient
    # dx = grads_bc[(1, 0)]
    # plot_interp_graph(Xs, dx[:, 0], title="dx")




if __name__ == "__main__":
    setup_logging(debug=True)
    torch.manual_seed(1)

    true_pde()


