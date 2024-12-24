import torch
from torch import Tensor
from cprint import c_print
from codetiming import Timer

from pde.BaseU import UBase
from pde.graph_grid.graph_store import DerivGraph, Point, Deriv, T_Point
from pde.graph_grid.graph_store import P_Types as T, P_TimeTypes as TT
from pde.findiff.findiff_coeff import gen_multi_idx_tuple, calc_coeff
from pde.findiff.fin_deriv_calc import FinDerivCalcSPMV, NeumanBCCalc
from pde.graph_grid.U_graph import UTemp
from pde.graph_grid.graph_utils import plot_points
# class UTemp(UBase):
#     _Xs: Tensor   # [N_us_tot, 2]                # Coordinates of nodes
#     _us: Tensor   # [N_us_tot, N_component]                   # Value at node
#     deriv_calc: FinDerivCalcSPMV
#
#     def __init__(self, Xs, us, deriv_calc, pde_mask, grad_mask):
#         self._Xs = Xs
#         self._us = us
#         self.deriv_calc = deriv_calc
#         self.pde_mask = pde_mask
#         self.grad_mask = grad_mask
#
#     def reset(self):
#         self._us = torch.zeros_like(self._us)
#
#     def _cuda(self):
#         pass

class UGraphTime(UBase):
    """ Holder for graph structure. """
    setup_dict: dict[int, T_Point]  # [N_us_tot]                  # Dictionary of points

    _Xs: Tensor   # [N_us_tot, 2]                # Coordinates of nodes
    _us: Tensor   # [N_us_tot, N_component]                   # Value at node
    deriv_val: Tensor # [N_deriv_BC*N_component]            # Derivative values at nodes for BC

    pde_mask: Tensor  # [N_us_tot]                   # Where dU/dt is enforced. Bool
    grad_mask: Tensor  # [N_us_tot, N_component]                   # Nodes that need to be updated. Bool
    neumann_mask: Tensor  # [N_us_tot]                   # Mask for derivative BC nodes. Bool
    neumann_mode: bool    # True if there are derivative BCs.

    N_us_tot: int           # Total number of points
    N_us_grad: int          # Number of points that need fitting
    N_pdes: int             # Number of points to enforce PDEs (excl BC)
    N_component: int           # Number of vector components
    N_deriv: int         # Number of derivatives used
    N_update: int         # Number of us that need updating

    graphs: dict[tuple, DerivGraph] # [N_graphs]                  # Gradient graphs for each gradient type.
        # edge_index: torch.Tensor   # [2, num_edges]      # Edges between nodes
        # edge_coeff: torch.Tensor  # [num_edges]       # Finite diff coefficients for each edge
        # neighbors: list[Tensor]     # [N_us_tot, N_neigh]           # Neighborhood for each node

    deriv_calc: FinDerivCalcSPMV
    deriv_orders_bc: dict[int, Deriv]  # [N_deriv_BC, 2]     # Derivative order for each derivative BC

    row_perm: Tensor # [N_us_grad] # Permutation for Jacobian matrix

    def _check(self, setup_dict):
        """ Check problem is well specified """
        points = list(setup_dict.values())
        types = [p.point_type for p in points]
        assert types.count(T.Ghost) == types.count(T.NeumCentralBC), "Number of ghost points must equal central Neumann BC points."

        for idx, t_p in setup_dict.items():
            P_type = t_p.point_type
            assert P_type[0] == P_type[1]

    def __init__(self, setup_dict: dict[int, T_Point], N_component, grad_acc:int = 2, max_degree:int = 2, device="cpu"):
        """ Initialize the graph with a set of points.
            setup_dict: dict[node_id, Point]. Dictionary of each type of point
         """
        self.device = device
        self.N_us_tot = len(setup_dict)
        self.N_us_grad = len([P for P in setup_dict.values() if TT.NORMAL in P.point_type])
        self.N_pdes = self.N_us_grad #len([P for P in setup_dict.values() if TT.NORMAL in P.point_type])
        self.N_component = N_component

        self._check(setup_dict)
        # 1) Reorder points. Redefines node values.
        sorted_points = sorted(setup_dict.values(), key=lambda x: x.X[1])
        self.setup_dict = {i: point for i, point in enumerate(sorted_points)}

        # 1.1) Masks for points that need updating.
        grad_mask = [[TT.FIXED not in P_type for P_type in P_comps.point_type]
                          for P_comps in self.setup_dict.values()]
        self.grad_mask = torch.tensor(grad_mask, dtype=torch.bool)

        # 1.2) Masks for points updated using du/dt ODE
        grad_setup_dict = {}
        for idx, t_p in self.setup_dict.items():
            P_type = t_p.point_type
            assert P_type[0] == P_type[1]
            if P_type[0] == TT.NORMAL:
                grad_setup_dict[idx] = Point(T.GRAD, t_p.X, value=None)
            else:
                grad_setup_dict[idx] = Point(T.DirichBC, t_p.X, value=None)
        pde_mask = [(T.GRAD in P.point_type) for P in grad_setup_dict.values()]
        self.pde_mask = torch.tensor(pde_mask, dtype=torch.bool)


        self.N_update = self.grad_mask.sum().item()

        # 2) Compute finite difference stencils / graphs for points that require fitting
        # Each gradient type has its own stencil and graph.
        diff_degrees = gen_multi_idx_tuple(max_degree)[1:] # 0th order is just itself.
        self.graphs = {}
        for degree in diff_degrees:
            c_print(f"Generating graph for degree {degree}", color="black")
            with Timer(text="Time to solve: : {:.4f}"):
                edge_idx, fd_weights = calc_coeff(grad_setup_dict, grad_acc, degree)
                self.graphs[degree] = DerivGraph(edge_idx, fd_weights)

        self._Xs = torch.stack([point.X for point in self.setup_dict.values()]).to(torch.float32)
        self._us = torch.tensor([point.init_val for point in self.setup_dict.values()], dtype=torch.float32)

        # PDE is enforced on normal points.
        #self.pde_mask = torch.tensor([T.PDE in P.point_type for P in self.setup_dict.values()])
        # U requires gradient for normal or ghost points.
        # self.grad_mask = torch.tensor([T.GRAD in P.point_type  for P in self.setup_dict.values()])


        if device == "cuda":
            self._cuda()

        self.deriv_calc = FinDerivCalcSPMV(self.graphs, self.pde_mask, self.pde_mask, self.N_us_tot, self.N_component, device=self.device)
        self.N_deriv = self.deriv_calc.N_deriv

        self.neumann_mode = False

    def get_subgraph(self, components=None, all_grads=False) -> UTemp:
        """ Make copy of subgraph with only certain components. """
        if components is None:
            components = [i for i in range(self.N_component)]

        if all_grads:
            mask = torch.ones_like(self.pde_mask).bool()
        else:
            mask = self.grad_mask

        N_component = len(components)
        deriv_calc = FinDerivCalcSPMV(self.graphs, mask, mask, self.N_us_tot, N_component, device=self.device)
        us_clone = self._us[:, components].clone()
        subraph_copy = UTemp(self._Xs.clone(), us_clone, deriv_calc, self.pde_mask, self.grad_mask[:, components])

        return subraph_copy

    def reset(self):
        self._us = torch.zeros_like(self._us)

    def set_bc(self, dirich_bc=None, neuman_bc=None):
        """ Set boundary conditions. """
        if dirich_bc is not None:
            dirich_bc = dirich_bc.to(self.device, non_blocking=True)
            assert dirich_bc.sum() == self.N_dirich, "Dirichlet BC must match number of Dirichlet points."
            assert dirich_bc.sum() == self.N_dirich, "Dirichlet BC must match number of Dirichlet points."
            self._us[self.dirich_mask] = dirich_bc

        # if neuman_bc is not None:
        #     neuman_bc = neuman_bc.to(self.device, non_blocking=True)
        #     self.deriv_val = neuman_bc
        #     self.neumann_mask = torch.tensor([True for _ in range(len(neuman_bc))], device=self.device)

    def get_grads(self):
        grad_dict = self.deriv_calc.derivative(self._us)
        return grad_dict

    def get_us_grad(self) -> list[torch.Tensor]:
        """ Returns us that need fitting, as list for each component since shapes are different.
            return.shape: [N_component][N_us_grad[i], 1]
        """
        # print(f'{self.grad_mask.shape = }')


        us_fit = [(self._us[:, i][self.grad_mask[:, i]]).unsqueeze(-1) for i in range(self.N_component)]
        return us_fit
        # [print(us_fit[i].shape) for i in range(self.N_component)]
        # assert False

    def update_grid(self, deltas):
        raise NotImplementedError

    def set_grid(self, new_us):
        """
        Set grid to new values.
        new_us.shape = [N_us_grad_total]
        """
        self._us[self.grad_mask] = new_us

    def set_grid_irreg(self, new_us):
        """
        Set grid to new values, if grid is irregular accross components.
        new_us.shape = [N_component][N_us_grad_comp]
        """
        for i, us in enumerate(new_us):
            self._us[self.grad_mask[:, i], i] = us
        #self._us = new_us

    def _cuda(self):
        """ Move graph data to CUDA. """
        self._us = self._us.cuda(non_blocking=True)
        self._Xs = self._Xs.cuda(non_blocking=True)

        self.grad_mask = self.grad_mask.cuda(non_blocking=True)
        self.pde_mask = self.pde_mask.cuda(non_blocking=True)
        #self.dirich_mask = self.dirich_mask.cuda(non_blocking=True)
        #self.grad_mask = self.grad_mask.cuda(non_blocking=True)
        [graph.cuda() for graph in self.graphs.values()]

    def _cuda_bc(self):
        self.deriv_val = self.deriv_val.cuda(non_blocking=True)
        self.neumann_mask = self.neumann_mask.cuda(non_blocking=True)

