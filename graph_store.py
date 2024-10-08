import torch
from dataclasses import dataclass
from graph_utils import gen_perim, show_graph
from scipy.spatial import KDTree
from cprint import c_print

from findiff.findiff_coeff import fin_diff_weights, ConvergenceError

# Code for point type
P_Normal = 0
P_Boundary = 1
P_Ghost = 2
P_dict = {P_Normal: "Normal", P_Boundary: "Boundary", P_Ghost: "Ghost"}

@dataclass
class Point:
    def __init__(self, X: torch.Tensor, point_type: int, value=None, direction: str=None):
        """
        Value:  If normal, value = initial value. Standard PDE enforced on this point.
                If boundary, value = boundary value. Dirichlet BC enforced on this point.
                If ghost, value = derivative value. Direction is direction of normal derivative boundary. Neuman BCs enforced on this point.
        """
        self.point_type = point_type
        self.X = X
        self.value = value
        self.direction = direction

    def __repr__(self):
        if self.value is None:
            return f'\033[33mPoint:\n     X={self.X}, \n     Type={P_dict[self.point_type]})\n\033[0m'
        else:
            return f'\033[33mPoint:\n     X={self.X}, \n     Type={P_dict[self.point_type]}, \n     Value={self.value})\n\033[0m'


def calc_coeff(Xs_all: torch.Tensor, point_dict: dict[int, Point], diff_acc: int, diff_order: tuple[int, int]):
    """ Calculate neighbours and finite difference coefficients
    Xs_all: torch.Tensor [N_nodes, 2]. All nodes in the graph.
    point_dict: dict[int, Point]. Dictionary of points where gradients are calculated
    N_nodes: int
    diff_acc: int
    diff_order: Tuple[int, int]
    """
    N_nodes = len(Xs_all)
    kdtree = KDTree(Xs_all)

    edge_idxs, weights, neighbors = [], [], {}
    for j, point in point_dict.items():
        X = point.X
        if j % 100 == 0:
            c_print(f'Iteration {j}', color="bright_black")
        # Find the nearest neighbors and calculate coefficients.
        # If the calculation fails (not enough points for given accuracy), increase the number of neighbors until it succeeds.
        min_points = min(75, N_nodes)
        max_points = min(100, N_nodes + 1)
        for i in range(min_points, max_points, 10):
            try:
                d, neigh_idx = kdtree.query(X, k=i)
                neigh_Xs = Xs_all[neigh_idx]
                w, status = fin_diff_weights(X, neigh_Xs, diff_order, diff_acc, method="abs_weight_norm", atol=3e-4, eps=1e-7)
            except ConvergenceError as e:
                print(f"{j} Adding more points")
                continue
            else:
                break
        else:
            c_print(f"Using looser tolerance for point {j}, {X=}", color="bright_magenta")
            # Using Try again with looser tolerance, probably from fp64 -> fp32 rounding.
            try:
                _, neigh_idx = kdtree.query(X, k=100)
                neigh_Xs = Xs_all[neigh_idx]
                w, status = fin_diff_weights(X, neigh_Xs, diff_order, diff_acc, method="abs_weight_norm", atol=1e-3, eps=3e-7)
            except ConvergenceError as e:
                # Unable to find suitable weights.
                status, err_msg = e.args
                c_print(f'{i = }, {err_msg = }, {status = }', color='bright_magenta')

                # print(neigh_Xs)
                raise ConvergenceError(f'Could not find weights for {X.tolist()}') from None

        # Only create edge if weight is not 0
        mask = torch.abs(w) > 1e-5
        neigh_idx_want = torch.tensor(neigh_idx[mask])
        source_nodes = torch.full((len(neigh_idx_want),), j, dtype=torch.long)
        edge_idx = torch.stack([source_nodes, neigh_idx_want], dim=0)
        w_want = w[mask]

        edge_idxs.append(edge_idx)
        neighbors[j] = neigh_idx_want
        weights.append(w_want)


    edge_idxs = torch.cat(edge_idxs, dim=1)
    weights = torch.cat(weights)
    return edge_idxs, weights, neighbors


def main():
    from matplotlib import pyplot as plt

    torch.set_printoptions(precision=3, sci_mode=False)
    torch.random.manual_seed(2)
    points_bc = gen_perim(1, 1, 0.2)
    points_bc = [Point(X, P_Boundary, 0.) for X in points_bc]
    points_main = torch.rand([100, 2])
    points_main = [Point(X, P_Normal, 0.) for X in points_main]

    points_all = points_bc + points_main
    points_dict = {i: X for i, X in enumerate(points_all)}
    # print(f'{points_dict = }')

    edge_idx, weights, neighbors = calc_coeff(points_dict, 3, (2, 0))
    #
    # show_graph(edge_idx, points, torch.zeros_like(points[:, 0]))


if __name__ == "__main__":
    main()