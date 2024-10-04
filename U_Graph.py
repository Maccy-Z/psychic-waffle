from matplotlib import pyplot as plt
import torch
from torch import Tensor
from scipy.spatial import KDTree
from torch_geometric.data import Data

from findiff.findiff_coeff import fin_diff_weights, ConvergenceError

class UGraph:
    """ Holder for graph structure. """
    data: Data      # PyG holder for graph data.
    # data.Xs: torch.Tensor   # [N_nodes, 2]                # Coordinates of nodes, CPU only.
    # data.us: torch.Tensor   # [N_nodes]                   # Value at node
    # data.edge_index: torch.Tensor   # [2, num_edges]      # Edges between nodes
    # data.edge_coeff: torch.Tensor  # [num_edges, 1]       # Finite diff coefficients for each edge

    neighbors: list[Tensor]     # [N_nodes, N_neigh]           # Neighborhood for each node


    def __init__(self, Xs: Tensor, device="cpu"):
        """ Initialize the graph with a set of points.
            Xs.shape = [N_nodes, 2]
         """
        edge_idx, fd_weights, neighbors = self._calc_coeff(Xs)

        data = Data(edge_index=edge_idx,)
        data.us = test_fn(Xs).unsqueeze(-1) #torch.ones(Xs.shape[0], dtype=torch.float32)     # [N_nodes, 1]
        data.edge_coeff = fd_weights.unsqueeze(-1)                               # [N_edges]
        data.Xs = Xs


        self.data = data
        self.Xs = Xs
        self.neighbors = neighbors
        if device == "cuda":
            self._cuda()

    def _calc_coeff(self, Xs):
        """ Calculate neighbours and finite difference coefficients """
        kdtree = KDTree(Xs)
        diff_acc = 2
        diff_order = (2, 0)

        weights, neighbors = [], []
        resids = []
        for X in Xs:
            # Find the nearest neighbors and calculate coefficients.
            # If the calculation fails (not enough points for given accuracy), increase the number of neighbors until it succeeds.
            for i in range(25, 100, 5):
                try:
                    d, neigh_idx = kdtree.query(X, k=i)
                    neigh_Xs = Xs[neigh_idx]
                    w, status = fin_diff_weights(X, neigh_Xs, diff_order, diff_acc, method="abs_weight_norm")
                except ConvergenceError:
                    continue
                else:
                    resids.append(status['abs_res'])
                    # Only create edge if weight is not 0
                    mask = torch.abs(w) > 1e-5
                    neigh_idx_want = torch.tensor(neigh_idx[mask])
                    w_want = w[mask]
                    neighbors.append(neigh_idx_want)
                    weights.append(w_want)
                    break
            else:
                raise ConvergenceError(f'Could not find weights for {X}')

        # Construct edge_idx associated of graph.
        source_nodes = torch.cat(neighbors)
        dest_nodes = torch.cat([torch.full((len(dest_nodes),), i, dtype=torch.long)
                              for i, dest_nodes in enumerate(neighbors)])
        edge_idx = torch.stack([source_nodes, dest_nodes], dim=0)

        # Weights are concatenated.
        weights = torch.cat(weights)

        # resids = torch.stack(resids) * 100
        # print(f'{resids.mean() = }')
        return edge_idx, weights, neighbors

    def plot(self):
        """ Plot graph in 2D space with nearest neighbors. """
        Xs = self.data.Xs
        # Plot the points
        plt.figure(figsize=(8, 8))
        plt.scatter(Xs[:, 0], Xs[:, 1], c='blue', s=100, zorder=2)

        # Draw lines to the nearest neighbors
        for i, neighbors in enumerate(self.neighbors):
            for neighbor in neighbors:
                plt.plot([Xs[i, 0], Xs[neighbor, 0]], [Xs[i, 1], Xs[neighbor, 1]], 'k--', alpha=1 / (2 * i+1))
            # break

        # Set axis labels and title
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Nearest Neighbors for Each Node in 2D Space')

        # Show grid and plot
        plt.grid(True)
        plt.show()


    def _cuda(self):
        """ Move graph data to CUDA. """
        self.data.us = self.data.us.cuda(non_blocking=True)
        self.data.edge_index = self.data.edge_index.cuda(non_blocking=True)
        # self.data.edge_dist = self.data.edge_dist.cuda(non_blocking=True)

def test_fn(Xs):
    x, y = Xs[:, 0], Xs[:, 1]
    u = x ** 3  + y ** 3 + x + y
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
    points2 = torch.rand([50, 2]) * 1
    points = torch.cat((points, points2), dim=0)

    # Number of nearest neighbors to find
    u_graph = UGraph(points, device="cpu")

    #u_graph.plot()
    #show_graph(u_graph.data, value_name='us')

    us = u_graph.data.us
    edge_index = u_graph.data.edge_index
    edge_coeff = u_graph.data.edge_coeff
    grad_layer = FinDiffGrad()
    grads = grad_layer(us, edge_index, edge_coeff)

    grads = grads.squeeze()
    Xs = u_graph.data.Xs
    grad_true = grad_fn(Xs)
    for g, X, g_true in zip(grads, Xs, grad_true):
        x = X[0].item()
        g = g.item()
        print(f'{x = :.3g}, {g = :.3g}, {g_true = :.4g}')

    error = torch.abs(grads - grad_true).mean()
    print()

    print(f'{error = }')



if __name__ == "__main__":
    main()