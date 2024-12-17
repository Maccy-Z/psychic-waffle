import torch
from cprint import c_print
from sympy.integrals.heurisch import components

from pde.graph_grid.graph_store import Point, P_Types, Deriv
from pde.config import Config
from pde.time_dependent.U_time_graph import UGraphTime, UTemp
from pde.mesh_generation.subproc_gen_mesh import run_subprocess
from pde.time_dependent.time_cfg import ConfigTime
from pde.NeuralPDE_Graph import NeuralPDEGraph
from pde.graph_grid.U_graph import UGraph
from pde.graph_grid.graph_utils import plot_interp_graph, test_grid, gen_perim
from pde.pdes.PDEs import PressureNS

def mesh_graph(cfg):
    N_comp = 1

    deriv = [Deriv(comp=[0], orders=[(1, 0)], value=0.)]#, Deriv(comp=[1], orders=[(1, 0)], value=1.)]
    value = [0. for _ in range(N_comp)]
    # points, p_tags = gen_points_full()
    points, p_tags = run_subprocess()
    c_print(f'Number of mesh points: {len(points)}', "green")

    points = torch.from_numpy(points).float()
    Xs_all = {}
    for i, (point, tag) in enumerate(zip(points, p_tags)):
        value = [i/500 for _ in range(N_comp)]

        if P_Types.DERIV in tag:
            Xs_all[i] = Point(P_Types.Normal, point, value=value)
        else:

            Xs_all[i] = Point(tag, point, value=value)

    u_graph = UGraphTime(Xs_all, N_component=N_comp, grad_acc=4, device=cfg.DEVICE)

    with open("../pdes/save_u_graph.pth", "wb") as f:
        torch.save(u_graph, f)

    return u_graph

def new_graph(cfg):
    cfg = Config()
    N_comp = 3

    n_grid = 20
    spacing = 1/(n_grid + 1)

    deriv = [Deriv(comp=[0], orders=[(1, 0)], value=1.), Deriv(comp=[1], orders=[(1, 0)], value=0.)]

    Xs_perim = gen_perim(1, 1, spacing)
    perim_mask = (Xs_perim[:, 1] > 0) & (Xs_perim[:, 1] < 1) & (Xs_perim[:, 0] ==0)
    Xs_neumann = Xs_perim[perim_mask]
    #print(Xs_neumann)
    Xs_dirich = Xs_perim[~perim_mask]

    Xs_ghost = Xs_neumann.clone()
    Xs_ghost[:, 0] = Xs_ghost[:, 0] - spacing
    Xs_bulk = test_grid(spacing, (1- spacing), torch.tensor([n_grid, n_grid]), device="cpu")

    Xs_fix = [Point(P_Types.DirichBC, X, value=[1. for _ in range(N_comp)]) for X in Xs_dirich]
    #Xs_deriv = [Point(P_Types.NeumCentralBC , X, value=[0. for _ in range(N_comp)], derivatives=deriv) for X in Xs_neumann]
    Xs_deriv = [Point(P_Types.DirichBC, X, value=[0. for _ in range(N_comp)]) for X in Xs_neumann]
    Xs_ghost = [] # [Point(P_Types.Ghost, X, value=[0. for _ in range(N_comp)]) for X in Xs_ghost]
    Xs_bulk = [Point(P_Types.Normal, X, value= [0. for _ in range(N_comp)]) for X in Xs_bulk]


    Xs_all = {i: X for i, X in enumerate(Xs_deriv + Xs_fix + Xs_bulk + Xs_ghost)}
    u_graph = UGraphTime(Xs_all, N_component=N_comp, grad_acc=4, device=cfg.DEVICE)

    with open("save_u_graph.pth", "wb") as f:
        torch.save(u_graph, f)

    return u_graph

def load_graph(cfg):
    u_graph = torch.load("save_u_graph.pth", weights_only=False)
    return u_graph

class TimePDEFunc:
    def __init__(self, u_graph_main: UGraphTime, cfg: Config, cfg_T: ConfigTime):
        self.cfg = cfg
        self.cfg_T = cfg_T
        self.device = cfg.DEVICE
        self.dtype = torch.float32

        # Subgraph for intermediate state V_star
        self.v_star_graph = UGraph.from_time_graph(u_graph_main, components=[0, 1])

        # Solve pressure equation
        pde_fn_inner = PressureNS(cfg, device=cfg.DEVICE)
        p_graph = UGraph.from_time_graph(u_graph_main, components=[2])
        pde_solver = NeuralPDEGraph(pde_fn_inner, p_graph, cfg)
        self.pde_solver = pde_solver

        self.mu = 1
        self.rho = 1

    def solve(self, u_dus, Xs, t):
        """ u_dus: dict[degree, torch.Tensor]. shape = [N_deriv][N_us_grad, N_comp]
            Returns: Update to us at time t.
        """

        us = u_dus[(0, 0)]
        dudxs = torch.stack([u_dus[(1, 0)], u_dus[(0, 1)]], dim=1)      # shape = [N_us_grad, 2, 3]
        d2udx2s = torch.stack([u_dus[(2, 0)], u_dus[(0, 2)]], dim=1)
        # First two components are velocity vector.
        vs = us[..., :-1]
        dvdxs = dudxs[..., :-1]
        d2vdx2s = d2udx2s[..., :-1]
        # Last component is pressure
        ps = us[..., -1:]
        dpdxs = dudxs[..., -1]
        #d2pdx2s = d2udx2s[..., -1:]

        # Viscosity = 1/mu laplacian(v)
        viscosity = 1 / self.mu * d2udx2s.sum(dim=-1)     # shape = [N_us_grad, 2]

        # Convective = -(v . grad)v
        vs_expand = vs.unsqueeze(-1)        # shape = [N_us_grad, 2, 1]
        product = vs_expand * dvdxs         # shape = [N_us_grad, 2, 2]
        convective = - product.sum(dim=-1)    # shape = [N_us_grad, 2]

        # Pressure = -1/rho grad(p)
        pressure = -1 / self.rho * dpdxs

        # print(f'{viscosity.shape = }, {convective.shape = }, {pressure.shape = }')
        dv_star = viscosity + convective + pressure

        v_star = vs + self.cfg_T.dt * dv_star
        self.v_star_graph.set_grid(v_star)
        v_s_grad = self.v_star_graph.get_grads()
        div_v_s = v_s_grad[(1, 0)][..., 0] + v_s_grad[(0, 1)][..., 1]

        div_v_s += 3
        self.pde_solver.forward_solve(div_v_s)
        us, Xs = self.pde_solver.us_graph.get_all_us_Xs()
        print()
        print(f'{us.shape = }, {Xs.shape = }')
        plot_interp_graph(Xs, us[:, 0])
        #plot_interp_graph(Xs, us[:, 2])
        exit(9)
        return laplacian


class TimePDEBase:
    """ Have a main PDE U_graph that is updated with every t. For update:
        1) Clone U_graph.
        1.1) Clone U_graph if we want state to be saved for later
        2) Solve PDE with U_graph
        3) Update time-PDE with new values.

        Assume graph doesn't change so deriv calc and intermediate sparse caches can be kept.
        """
    cfg_T: ConfigTime
    cfg_in: Config

    PDE_timefn: TimePDEFunc

    u_graph_main: UGraphTime
    u_saves: dict[float, UTemp]

    def __init__(self, u_graph_main: UGraphTime, cfg_T: ConfigTime, cfg_in: Config):
        self.u_graph_main = u_graph_main
        self.cfg_T = cfg_T
        self.cfg_in = cfg_in
        self.u_saves = {}


        self.PDE_timefn = TimePDEFunc(u_graph_main, cfg_in, cfg_T)

        self.device = "cuda"
        self.dtype = torch.float32

    def solve(self):
        cfg_T = self.cfg_T

        timesteps = torch.linspace(cfg_T.time_domain[0], cfg_T.time_domain[1], cfg_T.timesteps * cfg_T.substeps, device=self.device, dtype=self.dtype)

        for step_num, t in enumerate(timesteps):
            if step_num % cfg_T.substeps == 0:
                self.u_saves[t.item()] = self.u_graph_main.get_subgraph()

            grads_dict = self.u_graph_main.get_grads()
            us_all, _ = self.u_graph_main.get_all_us_Xs()
            _, Xs = self.u_graph_main.get_us_Xs_pde()

            update = self.cfg_T.dt * self.PDE_timefn.solve(grads_dict, Xs, t)
            self.u_graph_main.update_grid(-update)


    def update_boundary(self):
        pass


def main():
    cfg = Config()
    time_cfg= ConfigTime()
    c_print(f'{time_cfg.dt = }', color="bright_magenta")

    #u_graph = new_graph(cfg)
    u_graph = load_graph(cfg)

    time_pde = TimePDEBase(u_graph, time_cfg, cfg)
    time_pde.solve()

    saved_graphs = time_pde.u_saves
    for t, graph in saved_graphs.items():
        us, Xs = graph.us, graph.Xs
        plot_interp_graph(Xs, us[:, 0], title=f"t={t :.4g}")


if __name__ == "__main__":
    main()