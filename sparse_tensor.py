import torch
from torch.autograd import Function

class SparseCSRTransposer:
    def __init__(self, csr_matrix, check_sparsity=False):
        """
        Transposer that transposes CSR matrices efficiently using a precomputed template.
        Strategy: Create CSR matrix with same sparsity, but entries are permutation index.
        Use COO to transpose the matrix and extract the row indices, and new permutation index.
        Finally, use the permutation index and new indices to construct the transposed matrix.

        Args:
            csr_matrix (torch.sparse_csr_tensor): A CSR matrix to extract the sparsity pattern.
            check_sparsity (bool): Whether to check if the input matrix has the same sparsity pattern. False saves memory.
        """
        self.check_sparsity = check_sparsity

        device = csr_matrix.device
        crow_indices = csr_matrix.crow_indices()
        col_indices = csr_matrix.col_indices()
        numel = len(col_indices)

        # Construct a second csr_matrix with same sparsity, but
        csr_temp = torch.sparse_csr_tensor(crow_indices, col_indices, torch.arange(numel, device=device) + 1, csr_matrix.size())
        csr_matrix_T = csr_temp.t().to_sparse_csr()

        self.crow_indices_T = csr_matrix_T.crow_indices().to(torch.int32)
        self.col_indices_T = csr_matrix_T.col_indices().to(torch.int32)
        self.perm_idx_T = csr_matrix_T.values() - 1
        self.size_T = (csr_matrix.size(1), csr_matrix.size(0))  # Transposed size

        if check_sparsity:
            self.crow_indices = crow_indices
            self.col_indices = col_indices
            self.numel = numel

    def transpose(self, csr_matrix):
        """
        Transpose a single CSR matrix using the precomputed template.
        """
        if self.check_sparsity:
            # Ensure the matrix has the same sparsity pattern
            crow_indices = csr_matrix.crow_indices()
            col_indices = csr_matrix.col_indices()
            numel = len(col_indices)
            assert numel == self.numel, "Matrix has different number of non-zero elements"
            assert torch.equal(crow_indices, self.crow_indices) and torch.equal(col_indices, self.col_indices), "Matrix has different sparsity pattern"

        # Permute values to transposed positions
        values = csr_matrix.values()
        values_T = values[self.perm_idx_T]

        # Create the transposed CSR tensor using the template
        A_T_csr = torch.sparse_csr_tensor(self.crow_indices_T, self.col_indices_T, values_T, size=self.size_T)

        return A_T_csr


def generate_random_sparse_matrix(rows, cols, density, device="cpu"):
    num_nonzeros = int(rows * cols * density)
    row_indices = torch.randint(0, rows, (num_nonzeros,))
    col_indices = torch.randint(0, cols, (num_nonzeros,))
    values = torch.randn(num_nonzeros)  # Random values for the non-zero entries

    edge_index = torch.stack([row_indices, col_indices], dim=0)
    return torch.sparse_coo_tensor(edge_index, values, (rows, cols)).to(device)


class SparseMatMul(Function):
    @staticmethod
    def forward(ctx, A, A_T, b):
        """
        Forward pass for sparse matrix-vector multiplication.
        Args:
            A (torch.sparse_csr_tensor): Sparse CSR matrix of shape (m, n).
            A_T (torch.sparse_csr_tensor): Transpose of A, shape (n, m).
            b (torch.Tensor): Dense vector of shape (n,).
        Returns:
            torch.Tensor: Resulting vector of shape (m,).
        """
        # Save the transposed sparse matrix for backward
        ctx.save_for_backward(A_T)

        # Perform sparse-dense matrix multiplication
        output = torch.mv(A, b)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass to compute gradient with respect to b.
        Args: grad_output (torch.Tensor): Gradient of the loss with respect to the output, shape (m,).
        Returns:
            Tuple[None, None, torch.Tensor]: Gradients with respect to inputs.
                                             None for A and A_T (since they are constants), and gradient with respect to b.
        """
        (A_T,) = ctx.saved_tensors

        # Compute gradient with respect to b: A^T * grad_output
        grad_b = torch.mv(A_T, grad_output) # torch.sparse.mm(A_T, grad_output.unsqueeze(1)).squeeze(1)

        return None, None, grad_b


class SparseMatMulOperator:
    def __init__(self, A):
        """
        Initialize the operator with a sparse CSR matrix A.
        Args: A (torch.sparse_csr_tensor): Sparse CSR matrix of shape (m, n).
        """
        # self.A = A
        self.A = torch.sparse_csr_tensor(A.crow_indices().to(torch.int32), A.col_indices().to(torch.int32), A.values(), A.size())
        # Compute the transpose once and cache it
        self.A_T = self.A.t().to_sparse_csr() # .coalesce_csr()


    def matmul(self, b):
        """
        Perform the multiplication A * b.
        Args: b (torch.Tensor): Dense vector of shape (n,).
        Returns: torch.Tensor: Resulting vector of shape (m,).
        """
        return SparseMatMul.apply(self.A, self.A_T, b)

    def __repr__(self):
        return f"SparseMatMulOperator with tensor: {self.A}"


# Example Usage
if __name__ == "__main__":
    import time

    torch.set_printoptions(precision=2, sci_mode=False)
    rows, cols = 100_000, 100_000
    density = 0.001

    A_coo = generate_random_sparse_matrix(rows, cols, density).cuda()
    A_csr = A_coo.to_sparse_csr()
    print("Original CSR Matrix A:")
    print()

    # Initialize the transposer with the first matrix
    print("Precomputing Transposer...")
    transposer = SparseCSRTransposer(A_csr, check_sparsity=False)
    print("Transposer Ready")

    # Create multiple CSR matrices with the same sparsity pattern but different values
    for i in range(5):
        print(i)
        # Generate different values for each matrix
        new_vals = torch.randn_like(A_csr.values())
        A_csr_new = torch.sparse_csr_tensor(A_csr.crow_indices(), A_csr.col_indices(), new_vals, A_csr.size()).cuda()

        torch.cuda.synchronize()
        st = time.time()
        A_t = transposer.transpose(A_csr_new)
        torch.cuda.synchronize()
        print(f"Time taken = {time.time() - st}")

        # A_t_coo = A_csr_new.to_sparse_coo().t()

        # print(coo_equal(A_t_coo, A_t.to_sparse_coo()))

