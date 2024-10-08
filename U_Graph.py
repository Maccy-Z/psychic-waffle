import torch
from torch import Tensor
from cprint import c_print

from graph_store import calc_coeff, Point, P_Boundary, P_Normal, P_Ghost
from findiff.findiff_coeff import gen_multi_idx_tuple




class UGraph:
    """ Holder for graph structure. """
    Xs_dict: dict[int, Point]  # [N_nodes, 2]                # Input properties of nodes.
    Xs: Tensor   # [N_nodes, 2]                # Coordinates of nodes, CPU only.
    us: Tensor   # [N_nodes]                   # Value at node
    pde_mask: Tensor  # [N_nodes]                   # Mask for nodes that need to be updated. Bool
    grad_mask: Tensor  # [N_nodes]                   # Mask for nodes that need to be updated. Bool

    graphs: dict[tuple, ...] # [N_graphs]                  # Gradient graphs for each gradient type.
        # edge_index: torch.Tensor   # [2, num_edges]      # Edges between nodes
        # edge_coeff: torch.Tensor  # [num_edges]       # Finite diff coefficients for each edge
        # neighbors: list[Tensor]     # [N_nodes, N_neigh]           # Neighborhood for each node

    adj_mat: list[Tensor]  # [N_nodes, N_neigh]           # Overall adjacency matrix of all graphs for jacobian calculation.

    def __init__(self, Xs_dict: dict[int, Point], grad_acc:int = 2, grad_degree:int = 2, device="cpu"):
        """ Initialize the graph with a set of points.
            Xs.shape = [N_nodes, 2]
         """
        self.Xs_dict = Xs_dict
        self.N_nodes = len(Xs_dict)

        self.us = torch.tensor([point.value for point in Xs_dict.values()])
        self.Xs = torch.stack([point.X for point in Xs_dict.values()])

        # PDE is enforced on normal points.
        self.pde_mask = torch.tensor([X.point_type == P_Normal for X in Xs_dict.values()])
        N_pde = self.pde_mask.sum().item()
        # Points that need u to be updated are either normal or ghost points.
        self.grad_mask = torch.tensor([X.point_type == P_Ghost or X.point_type == P_Normal for X in Xs_dict.values()])
        N_grad = self.grad_mask.sum().item()

        # Compute finite difference stencils / graphs.
        # Each gradient type has its own stencil and graph.
        diff_degrees = gen_multi_idx_tuple(grad_degree)[1:] # 0th order is just itself.
        # Only need gradients for points that have grad_mask=True.
        Xs_grad_dict = {i: X for i, X in Xs_dict.items() if self.grad_mask[i]}

        self.graphs = {}
        for degree in diff_degrees:
            c_print(f"Generating graph for degree {degree}", color="black")
            edge_idx, fd_weights, neighbors = calc_coeff(self.Xs, Xs_grad_dict, grad_acc, degree)
            self.graphs[degree] = [edge_idx, fd_weights, neighbors]


        # Create an overall adjacency matrix for jacobian calculation.
        adj_mat = []
        for point in range(self.N_nodes):
            neigh_all = []
            for _, _, neighs in self.graphs.values():
                neigh_all.append(neighs.get(point, torch.tensor([])))
            neigh_all = torch.cat(neigh_all)
            neigh_unique = torch.unique(neigh_all)
            adj_mat.append(neigh_unique)

        row_idxs = torch.cat([torch.full_like(col_idx, i) for i, col_idx in enumerate(adj_mat)])
        col_idxs = torch.cat(adj_mat)
        idxs = torch.stack([row_idxs, col_idxs], dim=0)
        values = torch.ones(idxs.shape[1])
        adj_mat_sp = torch.sparse_coo_tensor(idxs, values, (self.N_nodes, N_grad))

        if device == "cuda":
            self._cuda()


    def _cuda(self):
        """ Move graph data to CUDA. """
        self.data.us = self.data.us.cuda(non_blocking=True)
        self.data.edge_index = self.data.edge_index.cuda(non_blocking=True)
        # self.data.edge_dist = self.data.edge_dist.cuda(non_blocking=True)

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
    points_bc = [Point(X, P_Boundary, 0.) for X in points_bc]
    points_main = torch.rand([50, 2])
    points_main = [Point(X, P_Normal, 0.) for X in points_main]

    points_all = points_bc + points_main
    points_all = sorted(points_all, key=lambda x: x.X[0])
    points_dict = {i: X for i, X in enumerate(points_all)}

    # Number of nearest neighbors to find
    u_graph = UGraph(points_dict, device="cpu")

    show_graph(u_graph.graphs[(1, 0)][0], u_graph.Xs, u_graph.pde_mask)

    us = u_graph.us.unsqueeze(1)
    edge_idx = u_graph.edge_idx
    edge_coeff = u_graph.edge_coeff.unsqueeze(1)
    grad_layer = FinDiffGrad()

    grads = grad_layer(us, edge_idx, edge_coeff)

    grads = grads.squeeze()
    Xs = u_graph.Xs
    grad_true = grad_fn(Xs)

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