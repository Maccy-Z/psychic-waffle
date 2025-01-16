import torch
from cprint import c_print
import numpy as np

from pde.graph_grid.graph_store import Point,  Deriv, T_Point
from pde.graph_grid.graph_store import P_TimeTypes as TT, P_Types as T
from pde.config import Config
from pde.time_dependent.U_time_graph import UGraphTime, UTemp
from pde.mesh_generation.generate_mesh import gen_mesh_time
from pde.time_dependent.time_cfg import ConfigTime
from pde.NeuralPDE_Graph import NeuralPDEGraph
from pde.graph_grid.U_graph import UGraph
from pde.graph_grid.graph_utils import plot_interp_graph, test_grid, gen_perim, plot_points
from pde.pdes.PDEs import PressureNS

def mesh_graph(cfg):
    N_comp = 3


    xmin, xmax = 0, 3
    ymin, ymax = 0.0, 1.5
    Xs, p_tags = gen_mesh_time(xmin, xmax, ymin, ymax, areas=[5e-3, 10e-3])
    Xs = torch.from_numpy(Xs).float()
    c_print(f'Number of mesh points: {len(Xs)}', "green")

    # Set up PDE graph for pressure
    setup_pde = {}
    for i, (X, tag) in enumerate(zip(Xs, p_tags)):
        if tag == "Wall":
            deriv = [Deriv(comp=[0], orders=[(0, 1)], value=0.)]
            setup_pde[i] = Point(T.NeumOffsetBC, X, value=[0], derivatives=deriv)
        elif tag == "Left":
            deriv = [Deriv(comp=[0], orders=[(1, 0)], value=0.)]
            setup_pde[i] = Point(T.NeumOffsetBC, X, value=[0], derivatives=deriv)
            #setup_pde[i] = Point(T.Normal, X, value=[0])
        elif tag == "Right":
            setup_pde[i] = Point(T.DirichBC, X, value=[0])
        elif tag == "Normal" or tag == "Left_extra":
            setup_pde[i] = Point(T.Normal, X, value=[0])
        else:
            raise ValueError(f"Unknown tag {tag}")

    u_graph_pde = UGraph(setup_pde, N_component=1, grad_acc=2, device=cfg.DEVICE)

    with open("./save_u_graph.pth", "wb") as f:
        torch.save(u_graph_pde, f)

    return None, u_graph_pde


def load_graph(cfg):
    u_graph_T = torch.load("save_u_graph_T.pth", weights_only=False)
    u_graph_pde = torch.load("save_u_graph.pth", weights_only=False)
    return u_graph_T, u_graph_pde

class TimePDEFunc:
    def __init__(self, u_graph_T: UGraphTime, u_graph_PDE: UGraph, cfg: Config, cfg_T: ConfigTime):
        self.cfg = cfg
        self.cfg_T = cfg_T
        self.device = cfg.DEVICE
        self.dtype = torch.float32
        #self.u_graph_T = u_graph_T
        self.p_graph_PDE = u_graph_PDE

        # Subgraph for intermediate state V_star
        #self.v_star_graph = u_graph_T.get_subgraph(components=[0, 1], all_grads=True)

        # Solve pressure equation
        pde_fn_inner = PressureNS(cfg, device=cfg.DEVICE)

        self.P_solver = NeuralPDEGraph(pde_fn_inner, u_graph_PDE, cfg)

        self.mu = 1
        self.rho = 0.01

        # The PDE is solved on a different graph from time derivatives.
        self.P_pde_mask = u_graph_PDE.pde_mask
        self.P_grad_mask = u_graph_PDE.grad_mask
        #self.V_star_mask = self.v_star_graph.pde_mask
        self.p_graph_PDE.set_eval_deriv_calc(torch.ones_like(self.P_pde_mask).bool())


    def solve(self, u_dus, Xs, t):
        """ u_dus: dict[degree, torch.Tensor]. shape = [N_deriv][N_us_grad, N_comp]
            Returns: new values of velocity and pressure at variable nodes at time t.
        """
        self.p_graph_PDE.reset()
        _, _X = self.p_graph_PDE.get_all_us_Xs()
        deriv_calc = self.p_graph_PDE.deriv_calc_eval

        div_v_s = [-1 if x[0] < 5 else -1 for x in _X]
        div_v_s = torch.tensor(div_v_s, device=self.device, dtype=self.dtype).unsqueeze(1)[self.P_pde_mask]

        # Neumann boundary masks
        indicator = self.p_graph_PDE.neumann_mask.float().unsqueeze(-1)
        grad_I = deriv_calc.derivative(indicator)
        grad_I_x = grad_I[(1, 0)][self.P_pde_mask]
        grad_I_y = grad_I[(0, 1)][self.P_pde_mask]

        # Solving
        print(div_v_s.shape, grad_I_x.shape, grad_I_y.shape)
        self.P_solver.forward_solve((div_v_s, grad_I_x, grad_I_y))
        _p, _X = self.p_graph_PDE.get_all_us_Xs()
        # _p[self.p_graph_PDE.neumann_mask] = 0
        plot_interp_graph(_X, _p[:, 0], title=f"P new- Step {t}")

        # Plotting manual laplacian
        dP_grads = deriv_calc.derivative(_p)
        dPdx, dPdy = dP_grads[(1, 0)], dP_grads[(0, 1)]
        d2Pdx2 = deriv_calc.derivative(dPdx)[(1, 0)]
        d2Pdy2 = deriv_calc.derivative(dPdy)[(0, 1)]
        est_laplace = d2Pdx2 + d2Pdy2
        est_laplace = est_laplace.squeeze()
        est_laplace[~self.P_pde_mask] = torch.nan
        plot_interp_graph(_X, est_laplace, title=f"Manual laplacian- Step {t}")

        # Simple laplacian
        d2Pdx2_, d2Pdy2_ = dP_grads[(2, 0)], dP_grads[(0, 2)]
        simple_laplace = d2Pdx2_ + d2Pdy2_
        simple_laplace[~self.P_pde_mask] = torch.nan
        plot_interp_graph(_X, simple_laplace.squeeze(), title=f"simple_laplace- Step {t}")

        # # Auto laplacian
        # auto_laplace = dP_grads["laplacian"].squeeze()
        # auto_laplace[~self.P_pde_mask] = torch.nan
        # plot_interp_graph(_X, auto_laplace, title=f"Auto laplacian- Step {t}")

        # # Gradient
        dP_grads = deriv_calc.derivative(_p)
        grad_plot = dP_grads[(1, 0)]
        grad_plot[self.p_graph_PDE.neumann_mask] = torch.nan
        plot_interp_graph(_X, grad_plot.squeeze(), title=f"dPdx- Step {t}", lim=[-2, 2])
        # grad_plot = d2Pdx2 + d2Pdy2
        # grad_plot[~self.P_grad_mask] = torch.nan
        # plot_interp_graph(_X, grad_plot.squeeze(), title=f"laplacian- Step {t}")
        exit("DONE")
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

        timesteps = torch.linspace(cfg_T.time_domain[0], cfg_T.time_domain[1], cfg_T.timesteps * cfg_T.substeps, dtype=self.dtype)

        for step_num, t in enumerate(timesteps):
            print(f'{step_num = }, t = {t.item()}')

            #Plotting
            # us, Xs = self.u_graph_T.get_all_us_Xs()
            # plot_interp_graph(Xs, us[:, 0], title=f"Vx start of Step {step_num}")
            # plot_interp_graph(Xs, us[:, 1], title=f"Vy- Step {step_num}")
            # plot_interp_graph(Xs, us[:, 2], title=f"P- Step {step_num}")
            # exit(9)

            # if step_num % cfg_T.substeps == 0:
            #     self.u_saves[t] = self.u_graph_T.get_all_us_Xs()

            #_, Xs = self.u_graph_T.get_us_Xs_pde()

            update = self.PDE_timefn.solve(None, None, t)
            self.u_graph_T.set_grid_irreg(update)

            # us, Xs = self.u_graph_T.get_all_us_Xs()
            # plot_interp_graph(Xs, us[:, 0], title=f"Vx End of Step {step_num}")
            # plot_interp_graph(Xs, us[:, 2], title=f"Px End of Step {step_num}")
            #exit(8)

    def update_boundary(self):
        pass


def main():
    from pde.utils import setup_logging

    setup_logging(debug=True)

    cfg = Config()
    time_cfg= ConfigTime()
    c_print(f'{time_cfg.dt = }', color="bright_magenta")

    # u_g_T, u_g_PDE = load_graph(cfg)
    u_g_T, u_g_PDE = mesh_graph(cfg)

    time_pde = TimePDEBase(u_g_T, u_g_PDE , time_cfg, cfg)
    time_pde.solve()

    # saved_graphs = time_pde.u_saves
    # for t, graph in saved_graphs.items():
    #     us, Xs = graph.us, graph.Xs
    #     plot_interp_graph(Xs, us[:, 0], title=f"t={t :.4g}")


if __name__ == "__main__":
    main()