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
    Xs, p_tags = gen_mesh_time(xmin, xmax, ymin, ymax)
    Xs = torch.from_numpy(Xs).float()
    c_print(f'Number of mesh points: {len(Xs)}', "green")

    # Set up time-graph
    setup_T = {}
    for i, (X, tag) in enumerate(zip(Xs, p_tags)):
        if np.any(np.all(np.isclose([[xmax, ymin], [xmax, ymax]], X), axis=1)):

            #setup_T[i] = T_Point([TT.NORMAL, TT.NORMAL, TT.FIXED], X, init_val=value)
            continue
        elif tag == "Wall":
            value = [0 for _ in range(N_comp)]
            setup_T[i] = T_Point([TT.FIXED, TT.FIXED, TT.MANUAL], X, init_val=value)
        elif tag == "Left" or tag == "Left_extra":
            value = [1, 0, 0]
            setup_T[i] = T_Point([TT.FIXED, TT.FIXED, TT.MANUAL], X, init_val=value)
        elif tag == "Right":
            value = [0, 0, 0]
            setup_T[i] = T_Point([TT.NORMAL, TT.NORMAL, TT.FIXED], X, init_val=value)
        elif tag == "Normal":
            vx_init = 1 - (X[0]-0.5)/2.5
            value = [vx_init, 0, 0]

            setup_T[i] = T_Point([TT.NORMAL, TT.NORMAL, TT.NORMAL], X, init_val=value)
        else:
            raise ValueError(f"Unknown tag {tag}")

    u_graph_time = UGraphTime(setup_T, N_component=N_comp, grad_acc=3, device=cfg.DEVICE)
    # plot_points(u_graph_time._Xs, u_graph_time.grad_mask[:, 0], title="grad mask")
    # print(f'{ u_graph_time.grad_mask[:, 0].sum() = }')
    # exit(4)
    with open("./save_u_graph_T.pth", "wb") as f:
        torch.save(u_graph_time, f)


    # Set up PDE graph for pressure
    setup_pde = {}
    for i, (X, tag) in enumerate(zip(Xs, p_tags)):
        if np.any(np.all(np.isclose([[xmax, ymin], [xmax, ymax]], X), axis=1)):
            #setup_pde[i] = Point(T.DirichBC, X, value=[0])
            continue

        elif tag == "Wall":
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


    u_graph_pde = UGraph(setup_pde, N_component=1, grad_acc=3, device=cfg.DEVICE)
    # plot_points(u_graph_pde._Xs, u_graph_pde.grad_mask, title="grad mask")
    # plot_points(u_graph_pde._Xs, u_graph_pde.pde_mask, title="pde mask")
    # plot_points(u_graph_pde._Xs, u_graph_pde.neumann_mask, title="neum mask")
    #
    # exit(8)
    with open("./save_u_graph.pth", "wb") as f:
        torch.save(u_graph_pde, f)

    return u_graph_time, u_graph_pde


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
        self.u_graph_T = u_graph_T
        self.p_graph_PDE = u_graph_PDE

        # Subgraph for intermediate state V_star
        self.v_star_graph = u_graph_T.get_subgraph(components=[0, 1], all_grads=True)

        # Solve pressure equation
        pde_fn_inner = PressureNS(cfg, device=cfg.DEVICE)

        self.P_solver = NeuralPDEGraph(pde_fn_inner, u_graph_PDE, cfg)

        self.mu = 1
        self.rho = 1

        # The PDE is solved on a different graph from time derivatives. Some
        self.P_pde_mask = u_graph_PDE.pde_mask
        self.P_grad_mask = u_graph_PDE.grad_mask
        self.V_star_mask = self.v_star_graph.pde_mask
        u_graph_PDE.set_eval_deriv_calc(self.V_star_mask)

    def _v_star(self, dvdxs, d2vdx2s, dpdxs, vs):
        # Viscosity = 1/mu laplacian(v)
        viscosity = 1 / self.mu * d2vdx2s.sum(dim=-2)     # shape = [N_us_grad, 2]
        # Convective = -(v . grad)v
        vs_expand = vs.unsqueeze(-1)        # shape = [N_us_grad, 2, 1]
        product = vs_expand * dvdxs         # shape = [N_us_grad, 2, 2]
        convective = - product.sum(dim=-1)    # shape = [N_us_grad, 2]
        # Pressure = -1/rho grad(p)
        pressure = -1 / self.rho * dpdxs
        # Uncorrected velocity update
        dv_star =  pressure + convective + viscosity
        v_star = dv_star * self.cfg_T.dt + vs

        return v_star


    def solve(self, u_dus, Xs, t):
        """ u_dus: dict[degree, torch.Tensor]. shape = [N_deriv][N_us_grad, N_comp]
            Returns: new values of velocity and pressure at variable nodes at time t.
        """
        self.p_graph_PDE.reset()
        self.v_star_graph.reset()
        _, _X = self.p_graph_PDE.get_all_us_Xs()

        self.v_star_graph._us = self.u_graph_T._us[:, :2]
        us = u_dus[(0, 0)]

        dudxs = torch.stack([u_dus[(1, 0)], u_dus[(0, 1)]], dim=1)      # shape = [N_us_grad, 2, 3]
        d2udx2s = torch.stack([u_dus[(2, 0)], u_dus[(0, 2)]], dim=1)
        # First two components are velocity vector.
        vs = us[..., :-1]
        dvdxs = dudxs[..., :-1]
        d2vdx2s = d2udx2s[..., :-1]
        # Last component is pressure
        #ps = us[..., -1:]
        dpdxs = dudxs[..., -1]
        #d2pdx2s = d2udx2s[..., -1:]

        # V_star = V + dt * (1/mu laplacian(v) - (v . grad)v - 1/rho grad(p))
        v_star = self._v_star(dvdxs, d2vdx2s, dpdxs, vs)
        self.v_star_graph.set_grid(v_star)

        # Pressure correction: laplacian(dP) = rho/dt div(v_star)
        v_star_grad = self.v_star_graph.get_grads()
        div_v_s = v_star_grad[(1, 0)][..., 0] + v_star_grad[(0, 1)][..., 1]
        print(v_star.shape)
        print(div_v_s.shape)
        exit("3")

        # Solve pressure equation
        div_v_s[~self.V_star_mask] = 0
        div_v_s_ = div_v_s * self.rho / self.cfg_T.dt
        div_v_s_ = div_v_s_[self.P_pde_mask]
        self.P_solver.forward_solve(div_v_s_)

        # Update pressure
        dP = self.p_graph_PDE.get_us_grad()   # shape = [N_ps_grad, 1]
        P_old = self.u_graph_T.get_us_grad()[2]
        P_new = P_old + dP

        # Update velocity: V^{n+1} = V_star - dt/rho grad(dP)
        dP_grads = self.p_graph_PDE.get_eval_grads()
        grad_dP = torch.cat([dP_grads[(1, 0)], dP_grads[(0, 1)]], dim=1)
        v_new = v_star - self.cfg_T.dt / self.rho * grad_dP
        us_new = [v_new[:, 0], v_new[:, 1], P_new[:, 0]]

        """ Plotting """
        # torch.save({"grad_mask": self.P_grad_mask, "Xs": _X}, "graph.pth")
        if t > -0.0001:

            # Init divergence:
            div_v_s[~self.P_pde_mask] = 0
            div_v_s[0] = 0
            div_plot = div_v_s
            plot_interp_graph(_X, div_plot, title=f"Initial divergence Step {t:.3g}")

            # # Plot laplacian(dp) = div(v_star) error
            # resid = torch.zeros(_X.shape[0], device=self.device)
            # dP_grads = self.p_graph_PDE.get_grads()
            # laplacian = dP_grads[(2, 0)].squeeze() + dP_grads[(0, 2)].squeeze()
            # resid[self.P_pde_mask] = laplacian # - div_v_s_
            # resid = resid.abs()
            # plot_interp_graph(_X, resid, title=f"laplacian old Residual- Step {t:.3g}")

            # Laplacian new
            resid = torch.zeros(_X.shape[0], device=self.device)
            dP_grads = self.p_graph_PDE.get_grads()
            laplacian_new = dP_grads["laplacian"].squeeze()
            resid[self.P_pde_mask] = laplacian_new # - div_v_s_
            resid = resid.abs()
            plot_interp_graph(_X, resid, title=f"laplacian new- Step {t:.3g}")

            # # Laplacian diff
            # resid = torch.zeros(_X.shape[0], device=self.device)
            # dP_grads = self.p_graph_PDE.get_grads()
            # laplacian_new = dP_grads["laplacian"].squeeze()
            # laplacian_old = dP_grads[(2, 0)].squeeze() + dP_grads[(0, 2)].squeeze()
            # resid[self.P_pde_mask] = laplacian_new -laplacian_old
            # resid = resid.abs()
            # plot_interp_graph(_X, resid, title=f"laplacian diff- Step {t:.3g}")


            # P PDE residual
            resid = torch.zeros(_X.shape[0], device=self.device)
            P_resid = self.P_solver.newton_solver.residuals.abs()
            resid[self.P_grad_mask] = P_resid
            print(f'{_X.shape = }, {P_resid.shape = }')
            plot_interp_graph(_X, resid, title=f"PDE Residual- Step {t:.3g}")

            # self.v_star_graph.set_grid(v_star)
            # _u, _X = self.v_star_graph.get_all_us_Xs()
            # plot_interp_graph(_X, _u[:, 0], title=f"Vx star- Step {t}")
            # plot_interp_graph(_X, _u[:, 1], title=f"Vy star- Step {t}")

            # Plotting P
            self.p_graph_PDE.set_grid(P_new)
            _p, _X = self.p_graph_PDE.get_all_us_Xs()
            plot_interp_graph(_X, _p[:, 0], title=f"P new- Step {t}")
            #
            # # P grad
            # grad_dP_ = torch.zeros_like(_X)
            # dP_grads = self.p_graph_PDE.get_grads()
            # grad_dP = torch.cat([dP_grads[(1, 0)], dP_grads[(0, 1)]], dim=1)
            # grad_dP_[self.P_pde_mask, :] = grad_dP
            #
            # plot_interp_graph(_X, grad_dP_[:, 0], title=f"grad dP x- Step {t}")
            # plot_interp_graph(_X, grad_dP_[:, 1], title=f"grad dP y- Step {t}")

            #
            # #Plotting V
            self.v_star_graph.set_grid(v_new)
            _u, _ = self.v_star_graph.get_all_us_Xs()
            # plot_interp_graph(_X, _u[:, 0], title=f"Vx- Step {t:.3g}")
            # plot_interp_graph(_X, _u[:, 1], title=f"Vy- Step {t:.3g}")
            # #
            # Final divergence
            self.v_star_graph.set_grid(v_new)
            v_star_grad = self.v_star_graph.get_grads()
            div_v_s = v_star_grad[(1, 0)][..., 0] + v_star_grad[(0, 1)][..., 1]
            div_v_s[~self.V_star_mask] = 0
            div_v_s[~self.P_pde_mask] = 0
            plot_interp_graph(_X, div_v_s, title=f"Final divergence- Step {t}")
            exit(4)
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

            if step_num % cfg_T.substeps == 0:
                self.u_saves[t] = self.u_graph_T.get_all_us_Xs()

            grads_dict = self.u_graph_T.get_grads()
            _, Xs = self.u_graph_T.get_us_Xs_pde()

            update = self.PDE_timefn.solve(grads_dict, Xs, t)
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

    #u_graph = new_graph(cfg)
    u_g_T, u_g_PDE = load_graph(cfg)
    # u_g_T, u_g_PDE = mesh_graph(cfg)

    time_pde = TimePDEBase(u_g_T, u_g_PDE , time_cfg, cfg)
    time_pde.solve()

    # saved_graphs = time_pde.u_saves
    # for t, graph in saved_graphs.items():
    #     us, Xs = graph.us, graph.Xs
    #     plot_interp_graph(Xs, us[:, 0], title=f"t={t :.4g}")


if __name__ == "__main__":
    main()