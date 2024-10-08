import torch
from torch import Tensor
from cprint import c_print

from graph_utils import diag_permute, plot_sparsity, show_graph
from graph_store import calc_coeff, DerivGraph, Point, P_Boundary, P_Normal, P_Ghost
from findiff.findiff_coeff import gen_multi_idx_tuple
from sparse_tensor import SparseCSRTransposer



class UGraph:
    """ Holder for graph structure. """
    setup_dict: dict[int, Point]  # [N_nodes, 2]                # Input properties of nodes.
    Xs: Tensor   # [N_nodes, 2]                # Coordinates of nodes, CPU only.
    us: Tensor   # [N_nodes]                   # Value at node
    pde_mask: Tensor  # [N_nodes]                   # Mask for nodes that need to be updated. Bool
    grad_mask: Tensor  # [N_nodes]                   # Mask for nodes that need to be updated. Bool

    graphs: dict[tuple, DerivGraph] # [N_graphs]                  # Gradient graphs for each gradient type.
        # edge_index: torch.Tensor   # [2, num_edges]      # Edges between nodes
        # edge_coeff: torch.Tensor  # [num_edges]       # Finite diff coefficients for each edge
        # neighbors: list[Tensor]     # [N_nodes, N_neigh]           # Neighborhood for each node

    # adj_mat: list[Tensor]  # [N_nodes, N_neigh]           # Overall adjacency matrix of all graphs for jacobian calculation.

    def __init__(self, setup_dict: dict[int, Point], grad_acc:int = 2, grad_degree:int = 2, device="cpu"):
        """ Initialize the graph with a set of points.
            Xs.shape = [N_nodes, 2]
         """
        self.N_nodes = len(setup_dict)

        # Reorder points by order: P_Normal -> P_Ghost -> P_Boundary. Redefines node values.
        normal_points = {i: X for i, X in enumerate([P for P in setup_dict.values() if P.point_type == P_Normal]) }
        ghost_points = {i + len(normal_points): X for i, X in enumerate([P for P in setup_dict.values() if P.point_type == P_Ghost])}
        boundary_points = {i + len(normal_points) + len(ghost_points): X for i, X in enumerate([P for P in setup_dict.values() if P.point_type == P_Boundary])}
        setup_dict = {**normal_points, **ghost_points, **boundary_points}

        # Compute finite difference stencils / graphs.
        # Each gradient type has its own stencil and graph.
        diff_degrees = gen_multi_idx_tuple(grad_degree)[1:] # 0th order is just itself.
        self.graphs = {}
        for degree in diff_degrees:
            c_print(f"Generating graph for degree {degree}", color="black")
            edge_idx, fd_weights, neighbors = calc_coeff(setup_dict, grad_acc, degree)
            self.graphs[degree] = DerivGraph(edge_idx, fd_weights, neighbors)

        # Create an overall adjacency matrix for jacobian calculation.
        grad_mask = torch.tensor([X.point_type == P_Ghost or X.point_type == P_Normal for X in setup_dict.values()])
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

        # Permute the adjacency matrix to be as diagonal as possible.
        permute_idx = diag_permute(adj_mat_sp)
        permute_idx = torch.from_numpy(permute_idx.copy())

        # From here onwards, use permuted indices for everything.
        # Permute everything to new indices.
        perm_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(permute_idx)}
        self.edge_idx_jac = torch.tensor([perm_map[idx.item()] for idx in edge_idxs.flatten()]).reshape(edge_idxs.shape)
        for degree, graph in self.graphs.items():
            edge_idx = graph.edge_idx
            edge_idx = torch.tensor([perm_map[idx.item()] for idx in edge_idx.flatten()]).reshape(edge_idx.shape)
            graph.edge_idx = edge_idx
            graph.neighbors = None      # Dont need anymore.

        setup_dict = {perm_map[old_idx]: v for old_idx, v in setup_dict.items()}
        self.setup_dict = {k: setup_dict[k] for k in sorted(setup_dict.keys())}
        self.Xs = torch.stack([point.X for point in self.setup_dict.values()])
        self.us = torch.tensor([point.value for point in self.setup_dict.values()])
        # PDE is enfoced on normal points.
        self.pde_mask = torch.tensor([X.point_type == P_Normal for X in self.setup_dict.values()])
        # U requires gradient for normal or ghost points.
        self.grad_mask = grad_mask[permute_idx]

        # Precompute matrix transpose
        adj_mat_dummy = torch.sparse_coo_tensor(self.edge_idx_jac, torch.empty_like(self.edge_idx_jac[0]), (self.N_nodes, self.N_nodes))
        adj_mat_dummy = adj_mat_dummy.to_sparse_csr()
        self.transposer = SparseCSRTransposer(adj_mat_dummy)


        if device == "cuda":
            self._cuda()


    def _cuda(self):
        """ Move graph data to CUDA. """
        self.us = self.us.cuda(non_blocking=True)
        self.pde_mask = self.pde_mask.cuda(non_blocking=True)
        self.grad_mask = self.grad_mask.cuda(non_blocking=True)
        [graph.cuda() for graph in self.graphs.values()]

def test_fn(Xs):
    x, y = Xs[:, 0], Xs[:, 1]
    u = x**3 + y ** 3 + x + y
    return u
def grad_fn(Xs):
    x, y = Xs[:, 0], Xs[:, 1]
    grad_x = 6 * x
    return grad_x


def main():
    from graph_utils import show_graph, gen_perim
    from GNN_FinDiff import FinDiffGrad

    torch.set_printoptions(precision=3, sci_mode=False)
    torch.random.manual_seed(2)

    points_bc = gen_perim(1, 1, 0.2)
    points_bc = [Point(P_Boundary, X, 0.) for X in points_bc]
    points_main = torch.rand([25, 2])
    points_main = [Point(P_Normal, X, 0.) for X in points_main]

    points_all = points_main + points_bc
    points_all = sorted(points_all, key=lambda x: x.X[0])
    points_dict = {i: X for i, X in enumerate(points_all)}

    # Number of nearest neighbors to find
    u_graph = UGraph(points_dict, device="cpu")

    print(u_graph.graphs[(1, 0)].edge_idx.shape)
    print(u_graph.edge_idx_jac.shape)
    show_graph(u_graph.edge_idx_jac, u_graph.Xs, u_graph.pde_mask)
    show_graph(u_graph.graphs[(1, 0)].edge_idx, u_graph.Xs, u_graph.pde_mask)

    # us = u_graph.us.unsqueeze(1)
    # edge_idx = u_graph.edge_idx
    # edge_coeff = u_graph.edge_coeff.unsqueeze(1)
    # grad_layer = FinDiffGrad()
    #
    # grads = grad_layer(us, edge_idx, edge_coeff)
    #
    # grads = grads.squeeze()
    # Xs = u_graph.Xs
    # grad_true = grad_fn(Xs)

    # for g, X, g_true in zip(grads, Xs, grad_true):
    #     x = X[0].item()
    #     g = g.item()
    #     print(f'{x = :.3g}, {g = :.3g}, {g_true = :.4g}')

    # error = torch.abs(grads - grad_true).max().item()
    # print()
    # print(f'{error = }')
    #
    # D_mat = torch.sparse_coo_tensor(edge_idx, edge_coeff.squeeze(), (len(points), len(points))).T
    # D_mat = D_mat.coalesce()
    #
    # derivative = torch.sparse.mm(D_mat, us)
    # derivative = derivative.squeeze()
    #
    # error_spmm = torch.abs(derivative - grad_true).max().item()
    # print(error_spmm)




if __name__ == "__main__":
    main()