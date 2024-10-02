import torch
from typing import Literal

from findiff.min_norm import min_sq_norm, min_abs_norm

diff_options = Literal["pinv", "sq_weight_norm", "abs_weight_norm"]
class ConvergenceError(Exception):
    pass

def generate_multi_indices(m):
    """
    Generate all multi-indices (alpha_x, alpha_y) for monomials up to degree m.

    Parameters:
    - m (int): The maximum total degree of the monomials.

    Returns:
    - torch.Tensor: A tensor of shape (M, 2) containing all multi-indices, where M is the number of monomials.
      Each row represents a multi-index [alpha_x, alpha_y].
    """
    indices = []
    for total_degree in range(m + 1):
        for alpha_x in range(total_degree + 1):
            alpha_y = total_degree - alpha_x
            indices.append([alpha_x, alpha_y])
    return torch.tensor(indices, dtype=torch.long)

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
    if idx.numel() > 0:
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

def fin_diff_weights(center, points, derivative_order, m, method: diff_options):
    """
    Compute the finite difference weights for approximating the specified derivative at the center point.

    Parameters:
    - center (tensor): The coordinates (x0, y0) of the center point.
    - points (tensor): List of (x, y) coordinates of  points.
    - derivative_order (tuple): A tuple (k_x, k_y) specifying the orders of the derivative with respect to x and y.
    - m (int): The maximum total degree of the monomials (order of accuracy).

    Returns:
    - torch.Tensor: A tensor of shape (N,) containing the weights w_i for the finite difference approximation.
    """
    # Step 1: Generate multi-indices for monomials up to degree m
    multi_indices = generate_multi_indices(m)  # Shape (M, 2)

    # Step 2: Compute the D vector containing derivatives of monomials at the center
    D = compute_D_vector(multi_indices, derivative_order)  # Shape (M,)

    # Step 3: Compute coordinate differences between  points and the center point
    deltas = points - center
    delta_x, delta_y = deltas[:, 0], deltas[:, 1]  # Shape (N,)

    # Step 4: Construct the matrix A by evaluating monomials at the coordinate differences
    A = construct_A_matrix(delta_x, delta_y, multi_indices)  # Shape (N, M)
    A_T = A.T  # Transpose to shape (M, N) for the linear system

    # Step 5: Solve the linear system A_T w = D.
    # Step 5.1: Optionally weight magnitude of w, for underdetermined system / extra points.
    weights = torch.norm(deltas, dim=1) ** 3
    # print(f'{weights = }')
    if method=="pinv":
        A_T_pinv = torch.linalg.pinv(A_T)  # Compute the pseudoinverse of A_T
        w = A_T_pinv @ D  # Compute the weights w (Shape: (N,))
        status = None
    elif method=="sq_weight_norm":
        w, status = min_sq_norm(A_T, weights, D)
    elif method == "abs_weight_norm":
        w, status = min_abs_norm(A_T, D, weights)
    else:
        exit("Method not implemented")

    err = torch.norm(A_T @ w - D)
    if err > 1e-4:
        raise ConvergenceError(f'Error too large: {err = }')

    sq_res = w.unsqueeze(0) @ torch.diag(weights) @ w
    abs_res = w.abs() @ weights


    return w, {'err': err, 'status': status, 'sq_res': sq_res, 'abs_res': abs_res}


def main(center, points, derivative_order, m):
    points = torch.tensor(points, dtype=torch.float32)

    # Define the center point coordinates
    center = torch.tensor(center, dtype=torch.float32)  # Center point

    # Compute the finite difference weights
    weights, _ = fin_diff_weights(center, points, derivative_order, m, method="abs_weight_norm")

    # Display the computed weights
    print()
    for w, p in zip(weights, points):
        print(f"{p.tolist()}: {w:.4f}")


# Example usage
if __name__ == "__main__":
    # Define the center point coordinates
    x0, y0 = 0.0, 0.0  # Center point

    # Define the coordinates of points
    points = [#(-1, 1), (0, 1), (1, 1),
              (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
              #(-1, -1), (0, -1), (1, -1)
              ]  # List of (x, y) tuples

    # Specify the derivative order to approximate (e.g., first derivative with respect to x)
    derivative_order = (1,0)  # (k_x, k_y)

    # Set the maximum total degree of the monomials (order of accuracy)
    m = 4   # Should be at least the sum of the derivative orders

    # Compute the finite difference weights
    main((x0, y0), points, derivative_order, m)
