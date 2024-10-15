import torch
from typing import Literal
from functools import lru_cache
from scipy.spatial import KDTree

from pde.findiff.min_norm import min_sq_norm, min_abs_norm
from pde.graph_grid.graph_store import Point, P_Types

diff_options = Literal["pinv", "sq_weight_norm", "abs_weight_norm"]

class ConvergenceError(Exception):
    pass

def gen_multi_idx_tuple(m):
    """
    Indicies in tuple form for dict indexing
    """
    indices = []
    for total_degree in range(m + 1):
        for alpha_x in range(total_degree + 1):
            alpha_y = total_degree - alpha_x
            indices.append((alpha_x, alpha_y))

    indices = sorted(indices, key=lambda x: (x[0] + x[1], -x[0]))
    return indices

@lru_cache(maxsize=5)
def generate_multi_indices(m):
    """
    Generate all multi-indices (alpha_x, alpha_y) for monomials up to degree m.
    m (int): The maximum total degree of the monomials.
    Returns: List shape (M, 2) containing all multi-indices, where M is the number of monomials.
      Each row represents a multi-index [alpha_x, alpha_y].
    """

    indices = []
    for total_degree in range(m + 1):
        for alpha_x in range(total_degree + 1):
            alpha_y = total_degree - alpha_x
            indices.append([alpha_x, alpha_y])
    return indices

def compute_D_vector(alpha, k):
    """
    Compute the vector D containing the derivatives of the monomials evaluated at the center point. One-hot coefficient.
    alpha: List of all derivative indices
    k: Derivative order to compute
    """
    k = torch.tensor(k)  # Shape (2,)

    # Identify monomials where alpha == k
    condition = (alpha[:, 0] == k[0]) & (alpha[:, 1] == k[1])

    D = torch.zeros(alpha.size(0), dtype=torch.float32)

    idx = condition.nonzero(as_tuple=True)[0]

    assert idx.numel() == 1, f'Wanted derivaitive order must be less than M'
    # Compute D_j = k_x! * k_y!
    D_value = torch.exp(torch.lgamma(k[0] + 1) + torch.lgamma(k[1] + 1))
    D[idx] = D_value

    return D

def construct_A_matrix(delta_x, delta_y, multi_indices):
    """
    Construct the matrix A by evaluating the monomials at the coordinate differences.
    Parameters:
    - delta_x (torch.Tensor): Tensor of shape (N,) containing the x-coordinate differences of  points.
    - delta_y (torch.Tensor): Tensor of shape (N,) containing the y-coordinate differences of  points.
    - multi_indices (torch.Tensor): Tensor of shape (M, 2) containing the multi-indices of the monomials.
    Returns:
    - torch.Tensor: The matrix A of shape (N, M), where each element A[i, j] = (delta_x[i])^(alpha_x[j]) * (delta_y[i])^(alpha_y[j]).
    """
    # Extract alpha_x and alpha_y from multi-indices and reshape for broadcasting
    alpha_x = multi_indices[:, 0].unsqueeze(0).float()  # Shape (1, M)
    alpha_y = multi_indices[:, 1].unsqueeze(0).float()  # Shape (1, M)

    # Reshape delta_x and delta_y for broadcasting
    delta_x = delta_x.unsqueeze(1)  # Shape (N, 1)
    delta_y = delta_y.unsqueeze(1)  # Shape (N, 1)

    # Evaluate monomials using broadcasting to compute A matrix
    A = (delta_x ** alpha_x) * (delta_y ** alpha_y)  # Shape (N, M)

    return A

def fin_diff_weights(center, points, derivative_order, m, method: diff_options, atol=1e-4, eps=1e-7):
    """
    Compute the finite difference weights for approximating the specified derivative at the center point.
    Parameters:
    - center (tensor): The coordinates (x0, y0) of the center point.
    - points (tensor): List of (x, y) coordinates of  points.
    - derivative_order (tuple): A tuple (k_x, k_y) specifying the orders of the derivative with respect to x and y.
    - m (int): The maximum total degree of the monomials (order of accuracy).
    - method (str): Method to compute the weights. Options: 'pinv', 'sq_weight_norm', 'abs_weight_norm'.
    - atol (float): Acceptable absolute solve max error.
    - eps (float): Floating point error scale.
    Returns:
    - torch.Tensor: A tensor of shape (N,) containing the weights w_i for the finite difference approximation.
    """
    # Step 1: Generate multi-indices for monomials up to degree m
    multi_indices = generate_multi_indices(m)  # Shape (M, 2)
    multi_indices = torch.tensor(multi_indices, dtype=torch.long)

    # Step 2: Compute the D vector containing derivatives of monomials at the center
    D = compute_D_vector(multi_indices, derivative_order)  # Shape (M,)

    # Step 3: Compute coordinate differences between  points and the center point
    deltas = points - center
    delta_x, delta_y = deltas[:, 0], deltas[:, 1]  # Shape (N,)

    # Step 4: Construct the matrix A by evaluating monomials at the coordinate differences
    A = construct_A_matrix(delta_x, delta_y, multi_indices)  # Shape (N, M)
    A_T = A.T  # Transpose to shape (M, N) for the linear system

    # Step 5: Solve the linear system A_T w = D.
    # Step 5.1: Weight magnitude of w, for underdetermined system / extra points. (Error formula)
    weights = 1 /(m + 1) * torch.norm(deltas, p=2, dim=1) ** (m + 1) + eps    # TODO: Why 2 better than 1?

    if method=="pinv":
        A_T_pinv = torch.linalg.pinv(A_T)  # Compute the pseudoinverse of A_T
        w = A_T_pinv @ D  # Compute the weights w (Shape: (N,))
        status = None
    elif method=="sq_weight_norm":
        w, status = min_sq_norm(A_T, weights, D)
    elif method == "abs_weight_norm":
        try:
            w, status = min_abs_norm(A_T, D, weights)
        except ValueError as e:
            status = e.args[0]
            raise ConvergenceError(status, f'') from None
    else:
        exit("Method not implemented")

    err = A_T @ w - D
    max_err = torch.abs(err).max()
    if max_err > atol:
        # print(w)
        raise ConvergenceError(status, f'Error too large: {max_err.item() = :.3g}')

    sq_res = w.unsqueeze(0) @ torch.diag(weights) @ w
    abs_res = w.abs() @ weights


    return w, {'err': err, 'status': status, 'sq_res': sq_res, 'abs_res': abs_res}

def calc_coeff(point_dict: dict[int, Point], diff_acc: int, diff_order: tuple[int, int]):
    """ Calculate neighbours and finite difference coefficients
    Xs_all: torch.Tensor [N_nodes, 2]. All nodes in the graph.
    point_dict: dict[int, Point]. Dictionary of points where gradients are calculated
    N_nodes: int
    diff_acc: int
    diff_order: Tuple[int, int]
    """
    want_type = {P_Types.NORMAL, P_Types.GHOST}
    Xs_all = torch.stack([point.X for point in point_dict.values()])
    N_nodes = len(point_dict)
    kdtree = KDTree(Xs_all)

    pde_dict = {idx: point for idx, point in point_dict.items() if point.point_type in want_type}

    edge_idxs, weights, neighbors = [], [], {}
    for j, point in pde_dict.items():
        X = point.X
        # Find the nearest neighbors and calculate coefficients.
        # If the calculation fails (not enough points for given accuracy), increase the number of neighbors until it succeeds.
        min_points = min(75, N_nodes)
        max_points = min(150, N_nodes + 1)
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
                _, neigh_idx = kdtree.query(X, k=min(150, N_nodes))
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
        edge_idx = torch.stack([neigh_idx_want, source_nodes], dim=0)
        w_want = w[mask]
        edge_idxs.append(edge_idx)
        neighbors[j] = neigh_idx_want
        weights.append(w_want)


    edge_idxs = torch.cat(edge_idxs, dim=1)
    weights = torch.cat(weights)
    return edge_idxs, weights, neighbors


def main(center, points, derivative_order, m, method: diff_options):
    points = torch.tensor(points, dtype=torch.float32)
    center = torch.tensor(center, dtype=torch.float32)

    # Compute the finite difference weights
    weights, status = fin_diff_weights(center, points, derivative_order, m, method=method, atol=3e-4, eps=1e-7)

    # # Display the computed weights
    # print()
    print(f'err = {status["err"]}', status['sq_res'], status['abs_res'])
    for w, p in zip(weights, points):
        if w.abs() > 1e-5:
            print(f"{p.tolist()}: {w:.3f}")


# Example usage
if __name__ == "__main__":
    torch.set_printoptions(precision=3, sci_mode=False)
    # Define the center point coordinates
    x0, y0 = [0.689, 0.076]  # Center point

    # Define the coordinates of points
    points = [[0.689, 0.076],
        [0.687, 0.072],
        [0.666, 0.031],
        [0.754, 0.073],
        [0.614, 0.057],
        # [0.712, 0.205],
        # [0.548, 0.033],
        # [0.830, 0.142],
        # [0.642, 0.229],
        # [0.541, 0.011],
        # [0.717, 0.276],
        # [0.720, 0.308],
        # [0.443, 0.096],
        # [0.452, 0.149],
        # [0.525, 0.271],
        # [0.476, 0.220],
        # [0.964, 0.138],
        # [0.615, 0.381],
        # [0.393, 0.184],
        # [1.000, 0.000],
        # [0.996, 0.196],
        # [0.389, 0.226],
        # [0.343, 0.037],
        # [0.607, 0.419],
        # [0.332, 0.073],
        # [0.823, 0.407],
        # [0.296, 0.054],
        # [0.339, 0.260],
        # [0.637, 0.474],
        # [0.925, 0.439],
        # [0.251, 0.106],
        # [0.712, 0.522],
        # [0.500, 0.500],
        # [0.566, 0.533],
        # [0.807, 0.541],
        # [0.553, 0.538],
        # [0.987, 0.462],
        # [0.944, 0.492],
        # [0.217, 0.257],
        # [0.317, 0.419],
        # [0.378, 0.478],
        # [0.189, 0.236],
        # [0.714, 0.619],
        # [0.481, 0.588],
        # [0.700, 0.636],
        # [0.277, 0.474],
        # [0.115, 0.188],
        # [0.194, 0.390],
        # [0.784, 0.658],
        # [0.158, 0.339],
        # [0.189, 0.406],
        # [0.452, 0.632],
        # [0.250, 0.500],
        # [0.824, 0.679],
        # [0.713, 0.694],
        # [0.206, 0.477],
        # [0.533, 0.707],
        # [0.046, 0.176],
        # [0.964, 0.679],
        # [0.871, 0.713],
        # [0.827, 0.730],
        # [0.216, 0.550],
        # [0.866, 0.723],
        # [0.030, 0.208],
        # [0.014, 0.062],
        # [0.599, 0.745],
        # [0.694, 0.753],
        # [0.453, 0.716],
        # [0.000, 0.000],
        # [0.947, 0.723],
        # [0.500, 0.750],
        # [0.799, 0.768],
        # [0.136, 0.511],
        # [0.323, 0.684],
        # [0.469, 0.771],
        # [0.859, 0.790],
        # [0.078, 0.496],
        # [0.024, 0.425],
        # [0.341, 0.744],
        # [0.618, 0.829],
        # [0.524, 0.821],
        # [0.767, 0.836],
        # [0.050, 0.496],
        # [0.065, 0.529],
        # [0.398, 0.791],
        # [0.171, 0.658],
        # [0.010, 0.466],
        # [0.161, 0.656],
        # [0.495, 0.852],
        # [0.703, 0.882],
        # [0.818, 0.876],
        # [0.460, 0.855],
        # [0.006, 0.576],
        # [0.322, 0.840],
        # [0.002, 0.621]
              ]  # List of (x, y) tuples

    # Specify the derivative order to approximate (e.g., first derivative with respect to x)
    derivative_order = (1,0)  # (k_x, k_y)

    # Set the maximum total degree of the monomials (order of accuracy)
    m = 1   # Should be at least the sum of the derivative orders

    # Compute the finite difference weights
    main((x0, y0), points, derivative_order, m, method="abs_weight_norm")
    from matplotlib import pyplot as plt
    plt.scatter(*zip(*points))
    plt.show()