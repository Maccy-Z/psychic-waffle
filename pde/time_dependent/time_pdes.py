import torch
from cprint import c_print

from pde.graph_grid.graph_store import Point,  Deriv, T_Point
from pde.graph_grid.graph_store import P_TimeTypes as TT, P_Types as T
from pde.config import Config
from pde.time_dependent.U_time_graph import UGraphTime, UTemp
from pde.mesh_generation.generate_mesh import gen_mesh_time
from pde.time_dependent.time_cfg import ConfigTime
from pde.NeuralPDE_Graph import NeuralPDEGraph
from pde.graph_grid.U_graph import UGraph
from pde.graph_grid.graph_utils import plot_interp_graph, plot_points, show_graph
from pde.pdes.PDEs import PressureNS

def mesh_graph(cfg):
    N_comp = 2

    xmin, xmax = 0, 3
    ymin, ymax = 0.0, 1.5
    Xs, p_tags = gen_mesh_time(xmin, xmax, ymin, ymax, areas=[3e-3, 8e-3])
    Xs = torch.from_numpy(Xs).float()
    c_print(f'Number of mesh points: {len(Xs)}', "green")

    # Set up time-graph
    setup_T = []
    for i, (X, tag) in enumerate(zip(Xs, p_tags)):
        if tag == "Wall":
            value = [0 for _ in range(N_comp)]
            setup_T.append(T_Point([TT.FIXED, TT.FIXED], X, init_val=value))
        elif tag == "Left": # or tag == "Left_extra":
            value = [1, 0]
            setup_T.append(T_Point([TT.FIXED, TT.FIXED], X, init_val=value))
        elif tag == "Right":
            value = [0, 0]
            deriv = [Deriv(comp=[0], orders=[(1, 0)], value=0.), Deriv(comp=[1], orders=[(1, 0)], value=0.)] # dux/dx = 0 and duy/dx = 0
            setup_T.append(T_Point([TT.EXIT, TT.EXIT], X, init_val=value, derivatives=deriv))
            # # Also add exit ghost nodes
            # _X = X.clone()
            # _X[0] += 0.05
            # setup_T.append(T_Point([TT.EXIT, TT.EXIT], _X, init_val=value))
        elif tag == "Normal"  or tag == "Left_extra" :
            vx_init = 1 - (X[0]-0.5)/2.5
            value = [vx_init, 0]

            setup_T.append(T_Point([TT.NORMAL, TT.NORMAL], X, init_val=value))
        else:
            raise ValueError(f"Unknown tag {tag}")

    setup_T = {i: point for i, point in enumerate(setup_T)}
    u_graph_time = UGraphTime(setup_T, N_component=N_comp, grad_acc=4, device=cfg.DEVICE)

    # plot_points(u_graph_time._Xs, u_graph_time.neumann_mask[:, 0], title="grad mask")
    bc_derivs = u_graph_time.set_bc()#.view(-1, 2)
    _us, _X = u_graph_time.get_all_us_Xs()
    plot_interp_graph(_X, _us[:, 0], title="Initial Vx")
    plot_vls = torch.zeros_like(_X)
    plot_vls[u_graph_time.neumann_mask] = bc_derivs
    plot_interp_graph(_X, plot_vls[:, 0], title="Derivative dVx/dx")
    print(bc_derivs)
    exit(4)
    with open("./save_u_graph_T.pth", "wb") as f:
        torch.save(u_graph_time, f)

    # Set up PDE graph for pressure
    setup_pde = {}
    for i, (X, tag) in enumerate(zip(Xs, p_tags)):
        # if np.any(np.all(np.isclose([[xmax, ymin], [xmax, ymax]], X), axis=1)):
        #     #setup_pde[i] = Point(T.DirichBC, X, value=[0])
        #     continue

        if tag == "Wall":
            deriv = [Deriv(comp=[0], orders=[(0, 1)], value=0.)]
            setup_pde[i] = Point(T.NeumOffsetBC, X, value=[0], derivatives=deriv)
        elif tag == "Left":
            deriv = [Deriv(comp=[0], orders=[(1, 0)], value=0.)]
            setup_pde[i] = Point(T.NeumOffsetBC, X, value=[0], derivatives=deriv)
            #setup_pde[i] = Point(T.Normal, X, value=[0])
        elif tag == "Right":
            setup_pde[i] = Point(T.DirichBC, X, value=[0])
        elif tag == "Normal"  or tag == "Left_extra":
            setup_pde[i] = Point(T.Normal, X, value=[0])
        # elif tag == "Left_extra":
        #     setup_pde[i] = Point(T.DirichBC, X, value=[0])
        else:
            raise ValueError(f"Unknown tag {tag}")


    u_graph_pde = UGraph(setup_pde, N_component=1, grad_acc=2, device=cfg.DEVICE)
    # plot_points(u_graph_pde._Xs, u_graph_pde.updt_mask, title="grad mask")
    # plot_points(u_graph_pde._Xs, u_graph_pde.pde_mask, title="pde mask")
    # plot_points(u_graph_pde._Xs, u_graph_pde.dirich_mask, title="neum mask")

    # print(f'{u_graph_pde.N_us_tot = }, {u_graph_time.N_us_tot = }')
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

        self.p_graph_PDE.set_eval_deriv_calc(torch.ones_like(u_graph_PDE.pde_mask).bool())

        # Subgraph for intermediate state V_star
        self.v_star_graph = u_graph_T.get_subgraph(all_grads=True)
        self.dp_graph = u_graph_PDE.copy() # u_graph_PDE.get_subgraph() #

        # Solve pressure equation
        pde_fn_inner = PressureNS(cfg, device=cfg.DEVICE)

        self.p_solver = NeuralPDEGraph(pde_fn_inner, self.dp_graph, cfg)

        self.mu = 1
        self.rho = 0.01

        # The PDE is solved on a different graph from time derivatives.
        self.P_pde_mask = u_graph_PDE.pde_mask
        self.P_updt_mask = u_graph_PDE.updt_mask
        self.V_star_mask = self.v_star_graph.pde_mask

        self.v_deriv_calc = u_graph_T.deriv_calc
        self.p_deriv_calc = u_graph_PDE.deriv_calc_eval

        self._set_grad_I()

    def _set_grad_I(self):
        indicator = 1 - self.p_graph_PDE.neumann_mask.float().unsqueeze(-1)
        grad_I = self.p_graph_PDE.deriv_calc_eval.derivative(indicator)
        grad_I_x = grad_I[(1, 0)][self.P_pde_mask]
        grad_I_y = grad_I[(0, 1)][self.P_pde_mask]
        self.grad_I = (grad_I_x, grad_I_y)

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
        dv_star = convective + viscosity + pressure
        v_star = vs + dv_star * self.cfg_T.dt

        return v_star

    def _get_derivs(self, vs, ps):
        """ Compute derivatives of velocity and pressure. """
        u_dus = self.v_deriv_calc.derivative(vs)
        vs = u_dus[0, 0]  # shape = [N_us_grad, 2]. Without boundary points
        dvdxs = torch.stack([u_dus[(1, 0)], u_dus[(0, 1)]], dim=1)
        d2vdx2s = torch.stack([u_dus[(2, 0)], u_dus[(0, 2)]], dim=1)

        p_dps = self.p_deriv_calc.derivative(ps)
        ps = p_dps[(0, 0)][self.V_star_mask, -1]  # shape = [N_us_grad, 1]
        dpdxs = torch.stack([p_dps[(1, 0)], p_dps[(0, 1)]], dim=2)[self.V_star_mask, 0]   # shape [N_us_grad, 2]
        d2pdx2s = torch.stack([p_dps[(2, 0)], p_dps[(0, 2)]], dim=2)[self.V_star_mask, 0]
        return vs, dvdxs, d2vdx2s, ps, dpdxs, d2pdx2s

    def solve(self, t):
        """ u_dus: dict[degree, torch.Tensor]. shape = [N_deriv][N_us_grad, N_comp]
            Returns: new values of velocity and pressure at variable nodes at time t.
        """
        #self.p_graph_PDE.reset()
        self.v_star_graph.reset()
        us_old, _ = self.u_graph_T.get_all_us_Xs()
        p_old, _X = self.p_graph_PDE.get_all_us_Xs()
        us_old, p_old = us_old.clone(), p_old.clone()
        p_new, dp = p_old.clone(), torch.zeros_like(p_old)

        p_deriv_calc = self.p_graph_PDE.deriv_calc_eval

        # Initialise v_star with bounadry conditions
        self.v_star_graph._us = self.u_graph_T._us.clone()

        # Compute momentum equation estimate
        vs, dvdxs, d2vdx2s, ps, dpdxs, _ = self._get_derivs(us_old, p_old)
        # V_star = V + dt * (1/mu laplacian(v) - (v . grad)v - 1/rho grad(p))
        v_star_updt = self._v_star(dvdxs, d2vdx2s, dpdxs, vs)
        self.v_star_graph.set_grid(v_star_updt)
        v_star = self.v_star_graph.get_all_us_Xs()[0]

        # Solve pressure equation multiple times
        for i in range(1):
            self.dp_graph.reset()
            self.v_star_graph.set_grid(v_star_updt)

            # Pressure correction: laplacian(dP) = rho/dt div(v_star)
            v_star_grad = self.v_star_graph.get_grads()
            div_v_s = v_star_grad[(1, 0)][..., 0] + v_star_grad[(0, 1)][..., 1]
            # Solve pressure equation
            div_v_s_ = div_v_s.clone()
            div_v_s_[~self.V_star_mask] = 0
            div_v_s_ = div_v_s_ * self.rho / self.cfg_T.dt
            div_v_s_ = div_v_s_[self.P_pde_mask]
            self.p_solver.forward_solve([div_v_s_, self.grad_I[0], self.grad_I[1]])

            # Update pressure
            dp_star = self.dp_graph.get_all_us_Xs()[0]   # shape = [N_ps, 1]
            dp += dp_star
            p_new += dp_star
            self.dp_graph.set_grid(dp[self.P_updt_mask])

            # Update velocity: V_star_n+1 = V_star_n - dt/rho grad(dP)
            dP_grads = p_deriv_calc.derivative(dp_star) #self.dp_graph.get_eval_grads()
            grad_dP = torch.cat([dP_grads[(1, 0)], dP_grads[(0, 1)]], dim=1) # [self.V_star_mask]
            v_star = - self.cfg_T.dt / self.rho * grad_dP + v_star#[self.V_star_mask]
            v_star_updt = v_star[self.V_star_mask]

        v_new_updt = v_star_updt

        # plot_interp_graph(_X, p_new, title=f"P new- Step {t}")
        p_new = p_new[self.P_updt_mask]
        self.p_graph_PDE.set_grid(p_new)

        us_new = [v_star_updt[:, 0], v_star_updt[:, 1]]

        """ Plotting """
        # torch.save({"updt_mask": self.P_updt_mask, "Xs": _X}, "graph.pth")
        if t > 0.019:
            plot_graph = self.v_star_graph
            _X = self.v_star_graph.get_all_us_Xs()[1]

            # plot_interp_graph(_X, indicator.squeeze(), title=f"Neumann mask- Step {t}")
            # exit(9)
            # plot_interp_graph(_X, self.V_star_mask, title=f"PDE mask- Step {t}")
            # # Init divergence:
            div_plot = torch.full_like(_X[:, 0], torch.nan)
            div_plot[self.P_pde_mask] = -div_v_s_
            plot_interp_graph(_X, div_plot, title=f"Initial divergence Step {t:.3g}")

            # # Laplacian new
            # resid = torch.full_like(_X[:, 0], torch.nan)
            # dP_grads = self.p_graph_PDE.get_grads()
            # laplacian_new = dP_grads["laplacian"].squeeze()
            # resid[self.P_pde_mask] = laplacian_new
            # resid = resid.abs()
            # plot_interp_graph(_X, resid, title=f"laplacian new- Step {t:.3g}")
            #
            # # Laplacian Old
            # resid = torch.full_like(_X[:, 0], torch.nan)
            # dP_grads = self.dp_graph.get_grads()
            # laplacian = dP_grads[(2, 0)].squeeze() + dP_grads[(0, 2)].squeeze()
            # resid[self.P_pde_mask] = laplacian
            # resid = resid.abs()
            # plot_interp_graph(_X, resid, title=f"laplacian old- Step {t:.3g}")

            # # P PDE residual
            # resid = torch.zeros(_X.shape[0], device=self.device)
            # P_resid = self.P_solver.newton_solver.residuals.abs()
            # resid[self.P_updt_mask] = P_resid
            # plot_interp_graph(_X, resid, title=f"PDE Residual- Step {t:.3g}")

            # self.v_star_graph.set_grid(v_star)
            # _u, _X = self.v_star_graph.get_all_us_Xs()
            # plot_interp_graph(_X, _u[:, 0], title=f"Vx star- Step {t}")
            # plot_interp_graph(_X, _u[:, 1], title=f"Vy star- Step {t}")

            # Plotting P
            self.p_graph_PDE.set_grid(p_new)
            _p, _X = self.p_graph_PDE.get_all_us_Xs()
            plot_interp_graph(_X, _p[:, 0], title=f"P new- Step {t}")
            # #
            # # P grad
            # dP_grads = self.p_graph_PDE.get_eval_grads()
            # grad_dP_ = torch.cat([dP_grads[(1, 0)], dP_grads[(0, 1)]], dim=1)
            # # grad_dP_[self.P_pde_mask] = grad_dP
            # plot_interp_graph(_X, grad_dP_[:, 0], title=f"grad dP_x- Step {t}")
            # plot_interp_graph(_X, grad_dP_[:, 1], title=f"grad dP_y- Step {t}")

            # # Plotting manual laplacian
            # dP_grads = self.p_graph_PDE.get_eval_grads()
            # dPdx, dPdy = dP_grads[(1, 0)], dP_grads[(0, 1)]
            # dPdxdy = torch.cat([dPdx, dPdy], dim=1)
            # d2Pdx2 = self.p_graph_PDE.deriv_calc_eval.derivative(dPdx)[(1, 0)]
            # d2Pdy2 = self.p_graph_PDE.deriv_calc_eval.derivative(dPdy)[(0, 1)]
            # est_laplace = d2Pdx2 +d2Pdy2 #+
            # est_laplace = est_laplace.squeeze()
            # est_laplace[~self.P_pde_mask] = torch.nan
            # plot_interp_graph(_X, est_laplace, title=f"Estimated laplacian- Step {t}")

            # plot_interp_graph(_X, div_v_s_est, title=f"Estimated divergence- Step {t}")
            # plot_interp_graph(_X, est_laplace - true_laplace, title=f"Residual laplacian- Step {t}")
            # #Plotting V
            plot_graph.set_grid(v_new_updt)
            #self.v_star_graph._us = v_new
            _u, _ = plot_graph.get_all_us_Xs()
            plot_interp_graph(_X, _u[:, 0], title=f"Vx- Step {t:.3g}")
            plot_interp_graph(_X, _u[:, 1], title=f"Vy- Step {t:.3g}")
            # #
            # # Final divergence
            # self.v_star_graph.set_grid(v_new)
            # # v_star2 = self.v_star_graph.get_all_us_Xs()[0].clone()
            # # v_n_1 = v_star2 - 0.01*dPdxdy
            # # self.v_star_graph._us = v_n_1.clone()
            # _u, _ = self.v_star_graph.get_all_us_Xs()
            # plot_interp_graph(_X, _u[:, 0], title=f"Vx- Step {t:.3g}")
            # plot_interp_graph(_X, _u[:, 1], title=f"Vy- Step {t:.3g}")
            #
            # Final divergence
            plot_graph.set_grid(v_new_updt)
            #self.v_star_graph._us = v_new
            v_star_grad = plot_graph.get_grads()
            div_v_s = v_star_grad[(1, 0)][..., 0] + v_star_grad[(0, 1)][..., 1]
            div_v_s[~self.V_star_mask] = torch.nan
            # div_v_s[~self.P_pde_mask] = 0
            plot_interp_graph(_X, div_v_s, title=f"Final divergence- Step {t}", lim=[-2, 2])
            #
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

            # #Plotting
            # us, Xs = self.u_graph_T.get_all_us_Xs()
            # plot_interp_graph(Xs, us[:, 0], title=f"Vx start of Step {step_num}")
            # plot_interp_graph(Xs, us[:, 1], title=f"Vy- Step {step_num}")
            # exit(9)

            if step_num % cfg_T.substeps == 0:
                self.u_saves[t] = self.u_graph_T.get_all_us_Xs()

            update = self.PDE_timefn.solve(t)
            self.u_graph_T.set_grid_irreg(update)

            # us, Xs = self.u_graph_T.get_all_us_Xs()
            # plot_interp_graph(Xs, us[:, 0], title=f"Vx End of Step {step_num}")
            # exit(8)

    def update_boundary(self):
        pass


def main():
    from pde.utils import setup_logging

    setup_logging(debug=True)

    cfg = Config()
    time_cfg= ConfigTime()
    c_print(f'{time_cfg.dt = }', color="bright_magenta")

    #u_g_T, u_g_PDE = load_graph(cfg)
    u_g_T, u_g_PDE = mesh_graph(cfg)

    time_pde = TimePDEBase(u_g_T, u_g_PDE , time_cfg, cfg)
    time_pde.solve()

    # saved_graphs = time_pde.u_saves
    # for t, graph in saved_graphs.items():
    #     us, Xs = graph.us, graph.Xs
    #     plot_interp_graph(Xs, us[:, 0], title=f"t={t :.4g}")


if __name__ == "__main__":
    main()