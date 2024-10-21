import torch
from torch import Tensor
from cprint import c_print

from pde.BaseU import UBase
from pde.graph_grid.graph_utils import diag_permute
from pde.graph_grid.graph_store import DerivGraph, Point, P_Types
from pde.findiff.findiff_coeff import gen_multi_idx_tuple, calc_coeff
from pde.findiff.fin_deriv_calc import FinDerivCalcSPMV


class UGraph(UBase):
    """ Holder for graph structure. """
    setup_dict: dict[int, Point]  # [N_nodes, 2]                # Input properties of nodes.
    Xs: Tensor   # [N_nodes, 2]                # Coordinates of nodes
    us: Tensor   # [N_nodes]                   # Value at node

    pde_mask: Tensor  # [N_nodes]                   # Mask for where to enforce PDE on. Bool
    grad_mask: Tensor  # [N_nodes]                   # Mask for nodes that need to be updated. Bool

    graphs: dict[tuple, DerivGraph] # [N_graphs]                  # Gradient graphs for each gradient type.
        # edge_index: torch.Tensor   # [2, num_edges]      # Edges between nodes
        # edge_coeff: torch.Tensor  # [num_edges]       # Finite diff coefficients for each edge
        # neighbors: list[Tensor]     # [N_nodes, N_neigh]           # Neighborhood for each node

    deriv_calc: FinDerivCalcSPMV

    def __init__(self, setup_dict: dict[int, Point], grad_acc:int = 2, max_degree:int = 2, device="cpu"):
        """ Initialize the graph with a set of points.
            setup_dict: dict[node_id, Point]. Dictionary of each type of point
         """
        self.device = device
        self.N_nodes = len(setup_dict)

        # 1) Reorder points by order: P_Normal -> P_Ghost -> P_Boundary. Redefines node values.
        normal_points = {i: X for i, X in enumerate([P for P in setup_dict.values() if P.point_type == P_Types.NORMAL]) }
        ghost_points = {i + len(normal_points): X for i, X in enumerate([P for P in setup_dict.values() if P.point_type == P_Types.GHOST])}
        boundary_points = {i + len(normal_points) + len(ghost_points): X for i, X in enumerate([P for P in setup_dict.values() if P.point_type == P_Types.BOUNDARY])}
        setup_dict = {**normal_points, **ghost_points, **boundary_points}

        # 2) Compute finite difference stencils / graphs.
        # Each gradient type has its own stencil and graph.
        diff_degrees = gen_multi_idx_tuple(max_degree)[1:] # 0th order is just itself.
        self.graphs = {}
        for degree in diff_degrees:
            c_print(f"Generating graph for degree {degree}", color="black")
            edge_idx, fd_weights, neighbors = calc_coeff(setup_dict, grad_acc, degree)
            self.graphs[degree] = DerivGraph(edge_idx, fd_weights, neighbors)

        # 3) Create an overall adjacency matrix for jacobian calculation.
        grad_mask = torch.tensor([X.point_type == P_Types.GHOST or X.point_type == P_Types.NORMAL for X in setup_dict.values()])
        adj_mat = []
        for point in range(self.N_nodes):
            neigh_all = []
            for graph in self.graphs.values():
                neighs = graph.neighbors
                neigh_all.append(neighs.get(point, torch.tensor([])))

            neigh_all.append(torch.tensor([point]))  # Add self to the list of neighbors.
            neigh_all = torch.cat(neigh_all)
            neigh_unique = torch.unique(neigh_all).to(torch.int64)
            adj_mat.append(neigh_unique)

        row_idxs = torch.cat([torch.full_like(col_idx, i) for i, col_idx in enumerate(adj_mat)])
        col_idxs = torch.cat(adj_mat)
        edge_idxs = torch.stack([row_idxs, col_idxs], dim=0)
        dummy_val = torch.ones(edge_idxs.shape[1])
        adj_mat_sp = torch.sparse_coo_tensor(edge_idxs, dummy_val, (self.N_nodes, self.N_nodes))

        # 4) Permute the adjacency matrix to be as diagonal as possible.
        permute_idx = diag_permute(adj_mat_sp)
        permute_idx = torch.from_numpy(permute_idx.copy())
        perm_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(permute_idx)}

        # 4.1) Permute everything to new indices.
        # From here onwards, use permuted indices for everything.
        for degree, graph in self.graphs.items():
            edge_idx = graph.edge_idx
            edge_idx = torch.tensor([perm_map[idx.item()] for idx in edge_idx.flatten()]).reshape(edge_idx.shape)
            graph.edge_idx = edge_idx
            graph.neighbors = None      # Dont need anymore.

        setup_dict = {perm_map[old_idx]: v for old_idx, v in setup_dict.items()}
        setup_dict = {k: setup_dict[k] for k in sorted(setup_dict.keys())}
        self.Xs = torch.stack([point.X for point in setup_dict.values()])
        self.us = torch.tensor([point.value for point in setup_dict.values()])
        # PDE is enfoced on normal points.
        self.pde_mask = torch.tensor([X.point_type == P_Types.NORMAL for X in setup_dict.values()])
        # U requires gradient for normal or ghost points.
        self.grad_mask = grad_mask[permute_idx]

        self.N_us_grad = self.grad_mask.sum().item()
        self.N_points = len(normal_points) + len(boundary_points)
        self.N_tot_points = len(setup_dict)


        if device == "cuda":
            self._cuda()

        self.deriv_calc = FinDerivCalcSPMV(self.graphs, self.pde_mask, self.grad_mask, self.N_points, device=self.device)

    def split(self, Ns):
        """ Split grid into subgrids. """
        pass

    def _cuda(self):
        """ Move graph data to CUDA. """
        self.us = self.us.cuda(non_blocking=True)
        self.Xs = self.Xs.cuda(non_blocking=True)
        self.pde_mask = self.pde_mask.cuda(non_blocking=True)
        self.grad_mask = self.grad_mask.cuda(non_blocking=True)
        [graph.cuda() for graph in self.graphs.values()]

def test_fn(Xs):
    x, y = Xs[0], Xs[1]
    u = x ** 2
    return u
def grad_fn(Xs):
    x, y = Xs[0], Xs[1]
    grad_x = 2 * x
    return grad_x


def main():
    from pde.graph_grid.graph_utils import show_graph, gen_perim
    from pde.findiff.fin_deriv_calc import FinDerivCalc, FinDerivCalcSPMV

    torch.set_printoptions(precision=3, sci_mode=False)
    torch.random.manual_seed(2)

    points_bc = gen_perim(1, 1, 0.5)
    points_bc = [Point(P_Types.BOUNDARY, X, test_fn(X)) for X in points_bc]
    points_main = torch.rand([5, 2])
    points_main = [Point(P_Types.NORMAL, X, test_fn(X)) for X in points_main]

    points_all = points_main + points_bc
    points_all = sorted(points_all, key=lambda x: x.X[0])
    points_dict = {i: X for i, X in enumerate(points_all)}

    # Number of nearest neighbors to find
    u_graph = UGraph(points_dict, grad_acc=2, max_degree=2, device="cpu")
    ORDER = (1, 0)

    # show_graph(u_graph.edge_idx_jac, u_graph.Xs, u_graph.pde_mask)
    show_graph(u_graph.graphs[ORDER].edge_idx, u_graph.Xs, u_graph.us)

    deriv_calc = FinDerivCalc(u_graph.graphs, u_graph.pde_mask)
    grads = deriv_calc.derivative(u_graph.us.unsqueeze(-1))


    """ Using sparse MV """
    deriv_calc_spmv = FinDerivCalcSPMV(u_graph.graphs, u_graph.pde_mask, u_graph.N_points)
    grads_spmv = deriv_calc_spmv.derivative(u_graph.us)

    Xs_grad = u_graph.Xs[u_graph.pde_mask]
    grad_true = torch.tensor([grad_fn(X) for X in Xs_grad])
    for g, g_true, X, g_spvm in zip(grads[ORDER], grad_true, Xs_grad, grads_spmv[ORDER]):
        g = g.item()
        print(f'{X[0] = :.3g}, {g_true = :.4g}, {g = :.4g}, {g_spvm = :.4g}')



if __name__ == "__main__":
    main()