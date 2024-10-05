from matplotlib import pyplot as plt
import torch
from torch import Tensor
from scipy.spatial import KDTree
from torch_geometric.data import Data
from cprint import c_print

from findiff.findiff_coeff import fin_diff_weights, ConvergenceError

class UGraph:
    """ Holder for graph structure. """
    Xs: torch.Tensor   # [N_nodes, 2]                # Coordinates of nodes, CPU only.
    us: torch.Tensor   # [N_nodes]                   # Value at node
    edge_index: torch.Tensor   # [2, num_edges]      # Edges between nodes
    edge_coeff: torch.Tensor  # [num_edges, 1]       # Finite diff coefficients for each edge

    neighbors: list[Tensor]     # [N_nodes, N_neigh]           # Neighborhood for each node


    def __init__(self, Xs: Tensor, device="cpu"):
        """ Initialize the graph with a set of points.
            Xs.shape = [N_nodes, 2]
         """
        self.N_nodes = Xs.shape[0]

        edge_idx, fd_weights, neighbors = self._calc_coeff(Xs)

        self.edge_idx = edge_idx
        self.edge_coeff = fd_weights
        self.us = test_fn(Xs)

        self.Xs = Xs
        self.neighbors = neighbors
        if device == "cuda":
            self._cuda()

    def _calc_coeff(self, Xs):
        """ Calculate neighbours and finite difference coefficients """
        kdtree = KDTree(Xs)
        diff_acc = 3
        diff_order = (2, 0)

        weights, neighbors = [], []
        for j, X in enumerate(Xs):
            if j % 100 == 0:
                c_print(f'Iteration {j}', color="bright_black")
            # Find the nearest neighbors and calculate coefficients.
            # If the calculation fails (not enough points for given accuracy), increase the number of neighbors until it succeeds.
            for i in range(75, 100, 10):
                try:
                    d, neigh_idx = kdtree.query(X, k=i)
                    neigh_Xs = Xs[neigh_idx]
                    w, status = fin_diff_weights(X, neigh_Xs, diff_order, diff_acc, method="abs_weight_norm", atol=3e-4, eps=1e-7)
                except ConvergenceError as e:
                    print(f"{j} Adding more points")
                    # torch.save(neigh_Xs, 'neigh_Xs.pt')
                    continue
                else:
                    break
            else:
                c_print(f"Using looser tolerance for point {j}, {X=}", color="bright_magenta")
                torch.save(neigh_Xs, 'neigh_Xs.pt')
                # print(neigh_Xs)
                # Using Try again with looser tolerance, probably from fp64 -> fp32 rounding.
                try:
                    _, neigh_idx = kdtree.query(X, k=100)
                    neigh_Xs = Xs[neigh_idx]
                    w, status = fin_diff_weights(X, neigh_Xs, diff_order, diff_acc, method="abs_weight_norm", atol=1e-3, eps=3e-7)
                    # c_print(f'{status = }', color='bright_blue')
                except ConvergenceError as e:
                    # Unable to find suitable weights.
                    status, err_msg = e.args
                    c_print(f'{i = }, {err_msg = }, {status = }', color='bright_magenta')

                    # print(neigh_Xs)
                    raise ConvergenceError(f'Could not find weights for {X.tolist()}') from None

            # Only create edge if weight is not 0
            mask = torch.abs(w) > 1e-5
            neigh_idx_want = torch.tensor(neigh_idx[mask])
            w_want = w[mask]
            neighbors.append(neigh_idx_want)
            weights.append(w_want)

        # Construct edge_idx associated of graph.
        source_nodes = torch.cat(neighbors)
        dest_nodes = torch.cat([torch.full((len(dest_nodes),), i, dtype=torch.long)
                              for i, dest_nodes in enumerate(neighbors)])
        edge_idx = torch.stack([source_nodes, dest_nodes], dim=0)

        # Weights are concatenated.
        weights = torch.cat(weights)

        return edge_idx, weights, neighbors


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
    from graph_utils import show_graph
    from GNN_FinDiff import FinDiffGrad

    torch.set_printoptions(precision=3, sci_mode=False)
    torch.random.manual_seed(2)
    # Example set of 2D points (shape: [num_nodes, 2])
    points = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],
        [0.5, .75],
        [.25, 0.5]
    ])
    points2 = torch.rand([10000, 2])
    points = torch.cat((points, points2), dim=0)
    points = points[points[:, 0].argsort()] * 2

    # Number of nearest neighbors to find
    u_graph = UGraph(points, device="cpu")

    #u_graph.plot()
    #show_graph(u_graph.edge_idx, u_graph.Xs, u_graph.us)

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

    error = torch.abs(grads - grad_true).max().item()
    print()
    print(f'{error = }')

    D_mat = torch.sparse_coo_tensor(edge_idx, edge_coeff.squeeze(), (len(points), len(points))).T
    D_mat = D_mat.coalesce()

    derivative = torch.sparse.mm(D_mat, us)
    derivative = derivative.squeeze()

    error_spmm = torch.abs(derivative - grad_true).max().item()
    print(error_spmm)




if __name__ == "__main__":
    main()