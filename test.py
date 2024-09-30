import torch


def moore_penrose_pinv(A, tol=1e-5):
    """
    Compute the Moore-Penrose inverse of matrix A using SVD.

    Args:
        A (torch.Tensor): The input matrix of shape (m, n).
        tol (float): Tolerance for singular values to be considered non-zero.

    Returns:
        torch.Tensor: The pseudo-inverse of matrix A of shape (n, m).
    """
    # Ensure A is a 2D tensor
    if A.dim() != 2:
        raise ValueError("Input matrix A must be 2-dimensional.")

    # Perform SVD
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)

    # Invert singular values with thresholding
    # S is a 1D tensor of singular values
    S_inv = torch.where(S > tol, 1.0 / S, torch.tensor(0.0, device=S.device))

    # Create a diagonal matrix for S_inv
    S_inv_mat = torch.diag(S_inv)

    # Compute the pseudo-inverse: V * S_inv * U^T
    A_pinv = Vh.T @ S_inv_mat @ U.T

    return A_pinv


# Example Usage
if __name__ == "__main__":
    # Create a sample matrix A
    A = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]], dtype=torch.float32)

    # Compute the pseudo-inverse manually
    A_pinv_manual = moore_penrose_pinv(A)

    # Compute the pseudo-inverse using PyTorch's built-in function for verification
    A_pinv_builtin = torch.linalg.pinv(A)

    # Display the results
    print("Manual Pseudo-Inverse:\n", A_pinv_manual)
    print("\nBuilt-in Pseudo-Inverse:\n", A_pinv_builtin)

    # Check if both pseudo-inverses are close
    print("\nAre both pseudo-inverses close? ", torch.allclose(A_pinv_manual, A_pinv_builtin, atol=1e-6))

    print(A @ A_pinv_manual @ A)