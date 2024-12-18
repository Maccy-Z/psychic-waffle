import torch
from cprint import c_print
from matplotlib import pyplot as plt

from pde.graph_grid.graph_store import Point,  Deriv, T_Point
from pde.graph_grid.graph_store import P_TimeTypes as TT, P_Types as T
from pde.config import Config
from pde.time_dependent.U_time_graph import UGraphTime, UTemp
#from pde.mesh_generation.subproc_gen_mesh import run_subprocess
from pde.mesh_generation.generate_mesh import gen_mesh_time

from pde.time_dependent.time_cfg import ConfigTime
from pde.NeuralPDE_Graph import NeuralPDEGraph
from pde.graph_grid.U_graph import UGraph
from pde.graph_grid.graph_utils import plot_interp_graph, test_grid, gen_perim
from pde.pdes.PDEs import PressureNS

def mesh_graph(cfg):
    N_comp = 3


    xmin, xmax = 0, 3
    ymin, ymax = 0.0, 1.5
    Xs, p_tags = gen_mesh_time(xmin, xmax, ymin, ymax)
    Xs = torch.from_numpy(Xs).float()
    c_print(f'Number of mesh points: {len(Xs)}', "green")

    # Set up time-graph
    setup_T = {}
    for i, (X, tag) in enumerate(zip(Xs, p_tags)):
        if tag == "Wall":
            value = [0 for _ in range(N_comp)]
            setup_T[i] = T_Point([TT.FIXED, TT.FIXED, TT.MANUAL], X, init_val=value)
        elif tag == "Left":
            value = [1, 0, 0]
            setup_T[i] = T_Point([TT.FIXED, TT.FIXED, TT.MANUAL], X, init_val=value)
        elif tag == "Right":
            value = [0, 0, 0]
            setup_T[i] = T_Point([TT.NORMAL, TT.NORMAL, TT.MANUAL], X, init_val=value)
        elif tag == "Normal":
            value = [0, 0, 0]
            setup_T[i] = T_Point([TT.NORMAL, TT.NORMAL, TT.NORMAL], X, init_val=value)

    u_graph_time = UGraphTime(setup_T, N_component=N_comp, grad_acc=4, device=cfg.DEVICE)
    #exit(4)
    with open("../pdes/save_u_graph_T.pth", "wb") as f:
        torch.save(u_graph_time, f)


    # Set up PDE graph for pressure
    deriv = [Deriv(comp=[0], orders=[(0, 1)], value=0.)]#, Deriv(comp=[1], orders=[(1, 0)], value=1.)]

    setup_pde = {}
    for i, (X, tag) in enumerate(zip(Xs, p_tags)):
        if tag == "Wall":
            deriv = [Deriv(comp=[0], orders=[(0, 1)], value=0.)]
            setup_pde[i] = Point(T.NeumOffsetBC, X, value=[0], derivatives=deriv)
        elif tag == "Left":
            deriv = [Deriv(comp=[0], orders=[(1, 0)], value=0.)]
            setup_pde[i] = Point(T.NeumOffsetBC, X, value=[0], derivatives=deriv)
        elif tag == "Right":
            setup_pde[i] = Point(T.DirichBC, X, value=[0])
        elif tag == "Normal":
            setup_pde[i] = Point(T.Normal, X, value=[0])

    u_graph_pde = UGraph(setup_pde, N_component=1, grad_acc=4, device=cfg.DEVICE)

    with open("../pdes/save_u_graph.pth", "wb") as f:
        torch.save(u_graph_pde, f)

    return u_graph_time, u_graph_pde

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

    Xs_fix = [Point(P_Types.DirichBC, X, value=[0 for _ in range(N_comp)]) for X in Xs_dirich]
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
    def __init__(self, u_graph_T: UGraphTime, u_graph_PDE: UGraph, cfg: Config, cfg_T: ConfigTime):
        self.cfg = cfg
        self.cfg_T = cfg_T
        self.device = cfg.DEVICE
        self.dtype = torch.float32

        # Subgraph for intermediate state V_star
        self.v_star_graph = u_graph_T.get_subgraph(components=[0, 1]) #UGraph.from_time_graph(u_graph_main, components=[0, 1])

        # Solve pressure equation
        pde_fn_inner = PressureNS(cfg, device=cfg.DEVICE)
        pressure_solver = NeuralPDEGraph(pde_fn_inner, u_graph_PDE, cfg)
        self.P_solver = pressure_solver

        self.mu = 1
        self.rho = 1

    def solve(self, u_dus, Xs, t):
        """ u_dus: dict[degree, torch.Tensor]. shape = [N_deriv][N_us_grad, N_comp]
            Returns: new values of velocity and pressure at variable nodes at time t.
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
        # Uncorrected velocity update
        dv_star = viscosity + convective + pressure
        v_star = vs + self.cfg_T.dt * dv_star
        self.v_star_graph.set_grid(v_star)

        # Pressure correction: laplacian(dP) = rho/dt div(v_star)
        v_star_grad = self.v_star_graph.get_grads()
        div_v_s = v_star_grad[(1, 0)][..., 0] + v_star_grad[(0, 1)][..., 1]
        div_v_s = div_v_s * self.rho / self.cfg_T.dt
        # Solve pressure equation
        self.P_solver.us_graph.reset()
        self.P_solver.forward_solve(div_v_s)
        dP = self.P_solver.us_graph.get_us_grad()
        dP_grads = self.P_solver.us_graph.get_grads()
        grad_dP = torch.cat([dP_grads[(1, 0)], dP_grads[(0, 1)]], dim=1)

        # Update pressure
        P_new = ps + dP
        # Update velocity: V^{n+1} = V_star - dt/rho grad(dP)
        v_new = v_star - self.cfg_T.dt / self.rho * grad_dP
        us_new = torch.cat([v_new, P_new], dim=1)


        return us_new


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

    def __init__(self, u_graph_T: UGraphTime, u_graph_PDE: UGraph, cfg_T: ConfigTime, cfg_in: Config):
        """ u_graph_T: Time graph.
            u_graph_PDE: Graph for internal PDE solver.
        """
        self.u_graph_T = u_graph_T
        self.u_graph_PDE = u_graph_PDE
        self.cfg_T = cfg_T
        self.cfg_in = cfg_in
        self.u_saves = {}


        self.PDE_timefn = TimePDEFunc(u_graph_T, u_graph_PDE, cfg_in, cfg_T)

        self.device = "cuda"
        self.dtype = torch.float32

    def solve(self):
        cfg_T = self.cfg_T

        timesteps = torch.linspace(cfg_T.time_domain[0], cfg_T.time_domain[1], cfg_T.timesteps * cfg_T.substeps, device=self.device, dtype=self.dtype)

        for step_num, t in enumerate(timesteps):
            print(f'{step_num = }, {t = }')
            if step_num % cfg_T.substeps == 0:
                self.u_saves[t.item()] = self.u_graph_T.get_subgraph()

            grads_dict = self.u_graph_T.get_grads()
            _, Xs = self.u_graph_T.get_us_Xs_pde()

            update = self.PDE_timefn.solve(grads_dict, Xs, t)
            self.u_graph_T.set_grid(update)

            # Plotting
            us, Xs = self.u_graph_T.get_all_us_Xs()
            plot_interp_graph(Xs, us[:, 0], title="Vx")
            plot_interp_graph(Xs, us[:, 1], title="Vy")

            plot_interp_graph(Xs, us[:, 2], title="P")


    def update_boundary(self):
        pass


def main():
    cfg = Config()
    time_cfg= ConfigTime()
    c_print(f'{time_cfg.dt = }', color="bright_magenta")

    u_g_T, u_g_PDE = mesh_graph(cfg)
    #u_graph = new_graph(cfg)
    #u_graph = load_graph(cfg)

    time_pde = TimePDEBase(u_g_T, u_g_PDE , time_cfg, cfg)
    time_pde.solve()

    saved_graphs = time_pde.u_saves
    for t, graph in saved_graphs.items():
        us, Xs = graph.us, graph.Xs
        plot_interp_graph(Xs, us[:, 0], title=f"t={t :.4g}")


if __name__ == "__main__":
    main()