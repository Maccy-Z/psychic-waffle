from pde.config import LinMode
import torch
from pde.solvers.linear_solvers import LinearSolver
from pde.utils_sparse import plot_sparsity, CSRPermuter
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
                # "preconditioner": "NOSOLVER",
                "preconditioner": {
                    "solver": "AMG",
                    "algorithm": "CLASSICAL",
                    #"selector": "SIZE_4",
                    "max_iters": 2,
                    "cycle": "V",
                    #"max_levels": 5,
                    #"max_matching_iterations": 1,
                }
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

    j = jacobian
    r = residuals
    print(sorted(torch.abs(jacobian.to_dense().diag()).tolist())[:100])
    # jacobian = jacobian.to_dense().to_sparse_csr()

    # jacobian = jacobian + torch.eye(jacobian.shape[0], device=jacobian.device)* 0.1

    # perm_mat, row_perm = random_permutation_matrix(jacobian.shape[0], dtype=jacobian.dtype, device=jacobian.device)
    # row_perm = torch.tensor(perm_dict["main"] + perm_dict["neum"])
    # c_print(f'{row_perm = }', color="bright_cyan")

    # row_perm = reverse_permutation(row_perm)
    # jacobian = j.to_dense()[row_perm]
    # residuals = r[row_perm]

    # permuter = CSRPermuter(row_perm, j)
    # jacobian = permuter.matrix_permute(j)
    # residuals = permuter.vector_permute(r)

    # print(torch.allclose(jac2.to_dense(), jacobian.to_dense()))
    plot_sparsity(jacobian)
    # plot_sparsity(jac2)

    # exit(9)
    # # jacobian = perm_mat@jacobian
    # # residuals = perm_mat@residuals


    # jac_dense = jacobian.to_dense()
    # print(f'{jac_dense.shape = }')
    # c_print(f'rank = {torch.linalg.matrix_rank(jac_dense)}', color='bright_cyan')
    # condition_num(jac_dense)
    # print()

    # AMGX solver
    solver = LinearSolver(LinMode.AMGX, "cuda", cfg=lin_solve_cfg)
    jacobian_amgx = solver.preproc_sparse(jacobian)
    result = solver.solve(jacobian_amgx, residuals)
    print(result[:25])

    # Exact solver
    solver_exact = LinearSolver(LinMode.SPARSE, "cuda")
    jacobian_cp = solver_exact.preproc(jacobian)
    true_result = solver_exact.solve(jacobian_cp, residuals)
    print(f'{true_result[:25]}')

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=200)
    main()

