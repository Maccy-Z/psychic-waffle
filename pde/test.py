from idlelib.debugger_r import restart_subprocess_debugger

from pde.config import LinMode
import torch
from pde.solvers.linear_solvers import LinearSolver
from pde.utils_sparse import plot_sparsity
from cprint import c_print

def condition_num(A):
    u, s, v = torch.svd(A)
    condition_number = s.max().item() / s.min().item()

    c_print(f'{condition_number = }', color="bright_cyan")


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
                #"print_solve_stats": 1,
                "solver": "PBICGSTAB",
                "convergence": "RELATIVE_INI_CORE",
                "tolerance": 1e-4,
                "max_iters": 1001,
                # "gmres_n_restart": 75,
                "preconditioner": "NOSOLVER",
                # "preconditioner": {
                #     "solver": "AMG",
                #     "algorithm": "AGGREGATION",
                #     "selector": "SIZE_4",
                #     "max_iters": 1,
                #     "cycle": "V",
                #     "max_levels": 5,
                #     "max_matching_iterations": 1,
                # }
            }
            # "solver": "BLOCK_JACOBI",
        }

def main():
    with open("jacobian.pth", "rb") as f:
        jacobian = torch.load(f)
    with open("residuals.pth", "rb") as f:
        residuals = torch.load(f)
    with open("jacob_pos.pth", "rb") as f:
        perm_dict = torch.load(f)

    jacobian = jacobian.to_dense().to_sparse_csr()

    #jacobian = jacobian + torch.eye(jacobian.shape[0], device=jacobian.device)* 0.1

    #perm_mat, row_perm = random_permutation_matrix(jacobian.shape[0], dtype=jacobian.dtype, device=jacobian.device)
    row_perm = torch.tensor(perm_dict["main"] + perm_dict["neum"])
    row_perm = reverse_permutation(row_perm)

    c_print(f'{row_perm = }', color="bright_cyan")

    jacobian = jacobian.to_dense()[row_perm]
    residuals = residuals[row_perm]
    #0jacobian = perm_mat@jacobian
    #residuals = perm_mat@residuals

    plot_sparsity(jacobian)

    jac_dense = jacobian.to_dense()
    print(f'{jac_dense.shape = }')
    print(f'rank = {torch.linalg.matrix_rank(jac_dense)}')
    condition_num(jac_dense)
    print()

    solver = LinearSolver(LinMode.AMGX, "cuda", cfg=lin_solve_cfg)
    jacobian = solver.preproc_sparse(jacobian)


    result = solver.solve(jacobian, residuals)

    print(result)

    true_result = torch.linalg.solve(jac_dense, residuals)
    print(f'{true_result = }')

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=200)
    main()

