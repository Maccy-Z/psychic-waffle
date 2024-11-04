from pde.config import LinMode
import torch
from pde.solvers.linear_solvers import LinearSolver
from pde.utils_sparse import plot_sparsity, CSRPermuter
from cprint import c_print



lin_solve_cfg = {
            "config_version": 2,
            "determinism_flag": 0,
            "exception_handling": 1,
            "solver": {"solver": "MULTICOLOR_DILU",
                       # "reorder_cols_by_color": 1,
                       # "insert_diag_while_reordering": 1,
                        "relaxation_factor": 0.5,
                       "max_iters": 1
                       }
    ,
}

def AMGX_solve(A, b):
    solver = LinearSolver(LinMode.AMGX, "cuda", cfg=lin_solve_cfg)
    A = solver.preproc_sparse(A)
    result = solver.solve(A, b)
    return result

def main():
    N = 50
    jacobian = torch.diag(torch.ones(N, device="cuda") * 1.5)
    jacobian = jacobian.diagonal_scatter(torch.ones(N-1), offset=1)
    jacobian = jacobian.diagonal_scatter(torch.ones(N-1), offset=-1)

    D = torch.diag(torch.diag(jacobian))
    L = torch.tril(jacobian, diagonal=-1)
    U = torch.triu(jacobian, diagonal=1)

    T = -torch.inverse(D) @ (L + U)
    eigenvalues = torch.abs(torch.linalg.eigvals(T))
    print(f'{eigenvalues = }')
    # exit(9)
    Xs = []
    for i in range(N):
        residual = torch.zeros(N, device="cuda")
        residual[i] = 1.
        x = AMGX_solve(jacobian, residual)
        Xs.append(x)

    Xs = torch.stack(Xs)
    A_inv = Xs  # Since B is identity matrix
    Xs_sparse = torch.abs(Xs) > 1e-3
    plot_sparsity(Xs_sparse)
    print(Xs)

    true_inv = torch.inverse(jacobian)
    c_print("True Inv", color="bright_cyan")
    c_print(f"{true_inv}", color="bright_cyan")

    # Test
    residual = torch.ones(N, device="cuda")
    x_test  = AMGX_solve(jacobian, residual)
    x_pred = A_inv @ residual
    x_true = true_inv @ residual
    #assert torch.allclose(x_test, x_pred, atol=1e-3)
    print(f'{x_test = }')
    #print(f'{x_pred = }')
    print(f'{x_true = }')

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(precision=2, sci_mode=False, linewidth=200)
    main()

