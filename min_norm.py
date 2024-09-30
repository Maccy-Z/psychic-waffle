import torch

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

    # Min val:
    L = c.T @ x
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

    """
    A_pinv = torch.linalg.pinv(A)       # Shape = (M, N)
    C = torch.diag(c)
    b = b.view(-1, 1)

    # Let u = A+ b
    u = A_pinv @ b

    # Let V = (I - A+ A)
    V = (torch.eye(A.shape[1]) - A_pinv @ A)

    # z = - (V^T C V)^-1 V^T C u
    v_c_v = V.T @ C @ V
    v_c_v_inv = torch.linalg.pinv(v_c_v)
    z = - v_c_v_inv @ V.T @ C @ u

    # Optimised x = u + Vz
    x = u + V @ z
    x = x.squeeze()

    return x

