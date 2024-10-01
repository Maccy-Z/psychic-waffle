import torch
import numpy as np

def print_tensor(tensor, threshold=1e-6):
    x = torch.where(torch.abs(tensor) < threshold, 0, tensor)
    print(x)

def min_norm_solve(A, c, b):
    """
    Solve Ax = b with min |c^T x|.
    :param A: Matrix (N, M)
    :param c: Vector (N)
    :param b: Vector (M)
    :return: x: Vector (N)

    In general x = A+ b + (I - A+ A)z, where A+ is the pseudo-inverse of A.
    Find Z first, then use for x

    """
    A_pinv = torch.linalg.pinv(A)       # Shape = (M, N)
    c = c.view(-1, 1)
    b = b.view(-1, 1)

    # Let s = c^T A+ b
    s = c.T @ A_pinv @ b

    # Let w = c^T(I - A+ A)
    w_T = c.T @ (torch.eye(A.shape[1]) - A_pinv @ A)
    w = w_T.T

    # z = s (w^T W w)^-1 W
    z = s / (w.T @ w) * w

    # x = A+ b + (I - A+ A)z
    x = A_pinv @ b - (torch.eye(A.shape[1]) - A_pinv @ A) @ z

    return x.squeeze()

def min_sq_norm(A, c, b):
    """
    Solve Ax = b with min x^T C x .
    :param A: Matrix (N, M)
    :param c: Vector (N)
    :param b: Vector (M)
    :return: x: Vector (N)

    In general x = A+ b + (I - A+ A)z, where A+ is the pseudo-inverse of A.
    Find Z first, then use for x
    z = - (V^T C V)^-1 V^T C u

    """

    A_pinv = torch.linalg.pinv(A, atol=1e-5)       # Shape = (M, N)
    # If A_pinv A is identity, return min norm solution
    if torch.allclose(A_pinv @ A, torch.eye(A.shape[1]), atol=1e-5):
        print("A_pinv A is identity")
        return A_pinv @ b

    C = torch.diag(c)
    b = b.view(-1, 1)

    # Let u = A+ b
    u = A_pinv @ b

    # Let V = (I - A+ A)
    # Use lstsq instead of pinv for better numerical stability
    A_inv_A = torch.linalg.lstsq(A, A).solution     # = A_pinv @ A
    V = (torch.eye(A.shape[1]) - A_inv_A)
    V[V.abs() < 1e-7] = 0     # If A is invertible, V should be zero

    # z = - (V^T C V)^-1 V^T C u
    v_c_v = V.T @ C @ V
    # v_c_v_inv = torch.linalg.pinv(v_c_v, atol=1e-5)
    z = - torch.linalg.lstsq(v_c_v, V.T @ C @ u).solution # = - v_c_v_inv @ V.T @ C @ u

    # Optimised x = u + Vz
    x = u + V @ z
    x = x.squeeze()

    # Solution is given by: x = C^-1 A^T (A C^-1 A^T)^-1 b
    # C_inv_A_T = torch.linalg.lstsq(C, A.T).solution # = C_inv @ A.T
    # A_C_inv_A_T = A @ C_inv_A_T
    # A_C_inv_A_T_inv = torch.linalg.pinv(A_C_inv_A_T)
    # x = C_inv_A_T @ A_C_inv_A_T_inv @ b

    return x.squeeze()


def min_abs_norm(A, b ,c):
    """ Solve min c^T |x| s.t. Ax = b """
    from scipy.optimize import linprog

    b_torch = b
    A, b, c = A.numpy(), b.numpy(), c.numpy()
    n = len(c)  # Number of variables

    # Objective function coefficients
    obj = np.concatenate((np.zeros(n), c))

    # Equality constraints
    A_eq = np.hstack((A, np.zeros((A.shape[0], n))))
    b_eq = b

    # Inequality constraints
    A_ub = np.vstack((
        np.hstack((-np.eye(n), -np.eye(n))),
        np.hstack((np.eye(n), -np.eye(n)))
    ))
    b_ub = np.zeros(2 * n)

    # Variable bounds
    bounds = [(-np.inf, np.inf)] * n + [(0, np.inf)] * n

    # Solve LP
    result = linprog(c=obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

    # Extract solution
    x = result.x[:n]
    return torch.tensor(x, dtype=b_torch.dtype, device=b_torch.device)

if __name__ == "__main__":
    A = torch.tensor([[1., 2, 4], [2, 5, 6]])  # Your matrix A
    b = torch.tensor([7., 8])  # Your vector b
    c = torch.tensor([0, 0, 1])   # Your vector c


    x1 = min_abs_norm(A, b, c)
    print('x1 =' )
    print_tensor(x1)
    print_tensor(A @ x1)

