from pde.config import LinMode
import torch
from pde.solvers.linear_solvers import LinearSolver
from pde.utils_sparse import plot_sparsity, CSRPermuter
from cprint import c_print
from pde.graph_grid.graph_utils import diag_permute

def condition_num(A):
    u, s, v = torch.svd(A)
    condition_number = s.max().item() / s.min().item()

    c_print(f'{condition_number = }', color="bright_cyan")

def power_iteration(A, num_iters=100, tol=1e-6, device='cpu'):
    """Estimate the largest singular value using Power Iteration."""
    # Initialize a random vector
    b_k = torch.randn(A.size(1), 1, device=device)
    b_k = b_k / torch.norm(b_k)

    for _ in range(num_iters):
        # Compute A^T * (A * b_k)
        Ab = torch.sparse.mm(A, b_k)
        AtAb = torch.sparse.mm(A.transpose(0, 1), Ab)

        # Compute the norm
        sigma = torch.norm(AtAb)

        # Re-normalize the vector
        b_k1 = AtAb / sigma

        # Check for convergence
        if torch.norm(b_k1 - b_k) < tol:
            break

        b_k = b_k1

    return sigma.item()

def inverse_power_iteration(A, num_iters=100, tol=1e-6, device='cpu'):
    """Estimate the smallest singular value using Inverse Power Iteration."""
    # Initialize a random vector
    b_k = torch.randn(A.size(1), 1, device=device)
    b_k = b_k / torch.norm(b_k)

    # To perform inverse iteration, we need to solve (A^T A) x = b_k
    # For large sparse matrices, consider using iterative solvers like CG
    # Here, we'll use a simple fixed number of iterations for demonstration

    for _ in range(num_iters):
        # Compute A^T * (A * b_k)
        AtAb = torch.sparse.mm(A.transpose(0, 1), torch.sparse.mm(A, b_k))

        # Solve (A^T A) x = b_k approximately
        # Here, using gradient descent step as a placeholder
        # For better performance, use a proper iterative solver
        x = AtAb.clone()  # Placeholder for solver step

        # Normalize
        b_k1 = x / torch.norm(x)

        # Check for convergence
        if torch.norm(b_k1 - b_k) < tol:
            break

        b_k = b_k1

    # After convergence, estimate the smallest singular value
    sigma_min = torch.norm(torch.sparse.mm(A, b_k))
    return sigma_min.item()

def condition_number(A, num_iters=100, tol=1e-6, device='cpu'):
    """Compute the condition number of a sparse matrix A."""
    sigma_max = power_iteration(A, num_iters, tol, device)
    sigma_min = inverse_power_iteration(A, num_iters, tol, device)
    return sigma_max / sigma_min if sigma_min != 0 else float('inf')



def reverse_permutation(indices):
    # Create an empty tensor for the reversed permutation with the same length as indices
    reversed_indices = torch.empty_like(indices)

    # Populate the reversed indices
    for i, target_position in enumerate(indices):
        reversed_indices[target_position] = i
    return reversed_indices

def random_permutation_matrix(n, dtype=torch.float32, device=None):
    """
    Generates a random permutation matrix of size n x n.

    A permutation matrix is a binary square matrix that has exactly one entry
    of 1 in each row and each column and 0s elsewhere.

    Args:
        n (int): The size of the permutation matrix (n x n).
        dtype (torch.dtype, optional): The desired data type of the returned tensor.
            Default is torch.float32.
        device (torch.device, optional): The device on which the tensor is to be allocated.
            If None, uses the current device.

    Returns:
        torch.Tensor: A random permutation matrix of shape (n, n).
    """
    # Generate a random permutation of indices from 0 to n-1
    perm = torch.randperm(n, device=device)

    # Create a one-hot encoded matrix based on the permutation
    perm_matrix = torch.eye(n, dtype=dtype, device=device)[perm]

    return perm_matrix, perm

lin_solve_cfg = {
            "config_version": 2,
            "determinism_flag": 0,
            "exception_handling": 1,
            "solver": {
                "monitor_residual": 1,
                "print_solve_stats": 1,
                "solver": "GMRES", #"PBICGSTAB", #
                "convergence": "RELATIVE_INI_CORE",
                "norm": "L2",
                "tolerance": 1e-4,
                "max_iters": 100,
                "gmres_n_restart": 100,
                "preconditioner": "NOSOLVER",
                #
                "preconditioner": {
                    "print_grid_stats": 1,
                    "smoother": {"solver": "JACOBI_L1",  # "MULTICOLOR_GS", #"BLOCK_JACOBI",#
                                 "relaxation_factor": 1.7,
                                 },
                    # "smoother": "NOSOLVER",
                    "solver": "AMG",
                    "coarse_solver": "DENSE_LU_SOLVER",
                    "algorithm": "AGGREGATION",  # "CLASSICAL", #
                    "selector": "SIZE_8",
                    "max_iters": 2,
                    "presweeps": 5,
                    "postsweeps": 5,
                    "cycle": "V",
                    "max_levels": 3,
                },
                # "preconditioner": {"solver": "BLOCK_JACOBI",
                #                     "relaxation_factor": 0.1,
                #                    "max_iters": 3
                #                    },
                }
            }

def main():
    with open("jacobian.pth", "rb") as f:
        jacobian = torch.load(f, weights_only=True)
    with open("residuals.pth", "rb") as f:
        residuals = torch.load(f, weights_only=True)
    with open("jacob_pos.pth", "rb") as f:
        perm_dict = torch.load(f, weights_only=True)
    N_jacob = int(jacobian.shape[0])

    residuals = torch.ones_like(residuals)
    #c_print(f'{residuals = }', color="bright_yellow")

    """Modify Jacobian"""
    # Permutation
    perm = diag_permute(jacobian.to_sparse_coo()).copy()
    # jacobian = jacobian.to_dense()[perm, :][:, perm].to_sparse_csr()
    # residuals = residuals[perm]

    # Normalise
    jac_dense = jacobian.to_dense()
    scale = torch.norm(jac_dense, dim=1)#
    #scale = torch.diag(jacobian.to_dense())
    mask = scale.abs() < 1.
    scale[mask] = 1.
    scale_mat = torch.diag(1 / scale)
    # jacobian = scale_mat @ jacobian
    # residuals = scale_mat @ residuals

    # print(condition_num(jac_dense))
    # exit(9)
    #plot_sparsity(jacobian)

    # AMGX solver
    solver = LinearSolver(LinMode.AMGX, "cuda", cfg=lin_solve_cfg)
    jacobian_amgx = solver.preproc_sparse(jacobian)
    result = solver.solve(jacobian_amgx, residuals)
    c_print(result[:25], color="bright_cyan")

    # Exact solver
    solver_exact = LinearSolver(LinMode.SPARSE, "cuda")
    jacobian_cp = solver_exact.preproc(jacobian)
    true_result = solver_exact.solve(jacobian_cp, residuals)
    c_print(f'{true_result[:25]}', color="cyan")

    resid = jacobian @ result - residuals
    resid_norm = torch.linalg.norm(resid)
    resid_percent = resid_norm / torch.linalg.norm(residuals) * 100
    c_print(f'Residual norm {resid_norm.item() :.3g}', color="bright_red")
    c_print(f'Residual {resid_percent.item() :.3g} %', color="bright_red")
    error_percent = torch.linalg.norm(result - true_result) / torch.linalg.norm(true_result) * 100
    c_print(f'Error percent: {error_percent.item():.3g}', color="bright_red")

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(precision=3, sci_mode=True, linewidth=200)
    main()

