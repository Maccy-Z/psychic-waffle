import torch
import time


def gmres(A, b, x0=None, tol=1e-5, max_iter=1000, restart=None):
    """
    GMRES algorithm to solve the linear system Ax = b.

    Arguments:
    A : callable or tensor
        If callable, it should be a function that returns the product Ax for a given x.
        If tensor, it should be a 2D sparse tensor representing the matrix A.
    b : tensor
        The right-hand side vector.
    x0 : tensor, optional
        Initial guess for the solution (default is None, which means x0 = 0).
    tol : float, optional
        Tolerance for convergence (default is 1e-5).
    max_iter : int, optional
        Maximum number of iterations (default is 1000).
    restart : int, optional
        Restart parameter (default is None, which means no restart).

    Returns:
    x : tensor
        Approximate solution to the system Ax = b.
    info : dict
        Dictionary containing information about the convergence (e.g., number of iterations).
    """
    device = b.device
    dtype = b.dtype
    n = b.size(0)

    A_in = A
    if x0 is None:
        x0 = torch.zeros_like(b)

    if restart is None:
        restart = n  # No restart if not specified

    def Ain_vec(v):
        return torch.mv(A_in, v)

    # Initialize
    x = x0

    r = b - Ain_vec(x)
    beta = torch.norm(r)
    eye = torch.eye(restart + 1, dtype=dtype, device=device)
    # if beta < tol:
    #     return x, {'converged': True, 'iterations': 0, 'residual_norm': beta.item()}

    Q = torch.zeros((n, restart + 1), dtype=dtype, device=device)
    H = torch.zeros((restart + 1, restart), dtype=dtype, device=device)

    Q[:, 0] = r / beta

    residual_norm = None
    for k in range(max_iter // restart):
        for j in range(restart):
            v = Ain_vec(Q[:, j])
            # Vectorized computation of H[0:j+1, j]
            H[:j+1, j] = torch.mv(Q[:, :j+1].T, v)
            # Vectorized update of v
            v -= torch.mv(Q[:, :j+1], H[:j+1, j])

            H[j+1, j] = torch.norm(v)
            # if H[j+1, j] < tol:
            #     break
            Q[:, j+1] = v / H[j+1, j]

        # Solve the least squares problem H * y = beta * e1
        eye[0] = beta
        y = torch.linalg.lstsq(H, eye)
        y = y.solution

        # Update the solution
        x += torch.mv(Q[:, :restart], y[:, 0])

        # Calculate the new residual
        r = b - Ain_vec(x)
        residual_norm = torch.norm(r)

        if residual_norm < tol:
            return x, {'converged': True, 'iterations': k * restart + j + 1, 'residual_norm': residual_norm}

        # Restart with new initial residual
        beta = residual_norm
        Q[:, 0] = r / beta
        H.zero_()

    return x, {'converged': False, 'iterations': max_iter, 'residual_norm': residual_norm}


def csr_mat_vec(data, col_idx, n_rows, row_indices, x):
    """
    Perform CSR sparse matrix times vector operation using PyTorch's scatter operation.

    Parameters:
    - data: 1D tensor of non-zero values in the matrix
    - col_idx: 1D tensor of column indices corresponding to the data tensor
    - crow_idx: 1D tensor that represents the row pointer array
    - x: 1D tensor, the vector to multiply the matrix with

    Returns:
    - result: 1D tensor, the result of the matrix-vector multiplication
    """

    result = torch.zeros(n_rows, device='cuda')

    # Get the rows indices for the non-zero values
    #row_indices = torch.arange(n_rows, device=data.device).repeat_interleave(crow_idx[1:] - crow_idx[:-1])

    # Compute the products of the non-zero elements with the corresponding elements in the vector
    products = data * x[col_idx] # x[col_idx]

    # Scatter add the products to the appropriate rows in the result vector
    result.scatter_add_(0, row_indices, products)
    # result = torch_scatter.scatter_sum(products, row_indices, dim=0)
    # result = torch.scatter_add(result, 0, row_indices, products)
    return result


def main():
    import numpy as np

    # # Example usage:
    # # Define a sparse matrix A and a vector b
    A = torch.load("A.pt").cuda()
    b = torch.load("b.pt").cuda()

    print(A.to_dense()[:5, :5])
    print()

    # Time default
    ts = []
    for _ in range(300):
        start = time.time()
        result1 = torch.mv(A, b)
        ts.append(time.time() - start)
    print(f'T = {100 * np.mean(ts[10:]) :.4g}')


    # Time my version
    data, col_idx, crow_idx = A.values(), A.col_indices(), A.crow_indices()
    n_rows = crow_idx.size(0) - 1
    row_indices = torch.arange(n_rows, device=data.device).repeat_interleave(crow_idx[1:] - crow_idx[:-1])
    ts = []
    for _ in range(300):
        start = time.time()
        result2 = csr_mat_vec(data, col_idx, n_rows, row_indices, b)
        ts.append(time.time() - start)
    print(f'{100 * np.mean(ts[10:]) :.4g}')

    print(f'All close: {torch.allclose(result1, result2)}')


if __name__ == "__main__":
    main()


