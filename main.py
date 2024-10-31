from pde.utils_sparse import gen_rand_sp_matrix, reverse_permutation
import torch


def permutation_to_sparse_csr(perm, dtype=torch.float32):
    """
    Convert a permutation tensor to a sparse permutation matrix in CSR format.

    Args:
        perm (torch.Tensor): 1D integer tensor representing the permutation.
                             It should contain each integer from 0 to n-1 exactly once.
        dtype (torch.dtype, optional): Data type of the sparse matrix values. Defaults to torch.float32.

    Returns:
        torch.Tensor: A sparse CSR tensor representing the permutation matrix.
    """
    n = perm.size(0)

    # Validate that perm is a valid permutation
    if not torch.all((perm >= 0) & (perm < n)):
        raise ValueError("Permutation tensor contains invalid indices.")
    if torch.unique(perm).numel() != n:
        raise ValueError("Permutation tensor must contain each index exactly once.")

    # Data is all ones
    data = torch.ones(n, dtype=dtype)

    # Indices are the permutation itself
    col_indices = perm.clone().to(torch.int32)

    # Indptr for CSR: [0, 1, 2, ..., n]
    row_ptr = torch.arange(n + 1, dtype=torch.int32)

    # Create sparse CSR tensor
    sparse_matrix = torch.sparse_csr_tensor(
        crow_indices=row_ptr,
        col_indices=col_indices,
        values=data,
        size=(n, n)
    )

    return sparse_matrix

class CSRPermuter:
    def __init__(self, perm_to, A_csr):
        """ Precompute matrix permutation.
            perm_to: 1D tensor. perm_to[IDX] = VALUE means row VALUE should be moved to IDX / where rows should be moved to.

        """
        A_crow_indices = A_csr.crow_indices()
        A_col_indices = A_csr.col_indices()
        A_values = A_csr.values()

        A_csr_permute = torch.sparse_csr_tensor(A_crow_indices, A_col_indices, torch.arange(len(A_values), dtype=torch.float64) + 1, A_csr.size())
        #permutation = reverse_permutation(permutation)
        perm_mat = permutation_to_sparse_csr(perm_to, dtype=torch.float64)
        out_mat = torch.sparse.mm(perm_mat, A_csr_permute)

        self.crow_indices = out_mat.crow_indices()
        self.col_indices = out_mat.col_indices()
        self.val_permutation = (out_mat.values() - 1).to(torch.int32)
        self.size = out_mat.size()

        self.perm_from = reverse_permutation(perm_to)

    def matrix_permute(self, A_csr):
        """ Precomputed permutation of a matrix."""
        A_values = A_csr.values()
        A_values_perm = A_values[self.val_permutation]

        A_permute = torch.sparse_csr_tensor(self.crow_indices, self.col_indices, A_values_perm, self.size)
        return A_permute

    def vector_permute(self, b):
        """ Precomputed permutation of a vector."""
        return b[self.perm_from]

def main():
    from cprint import c_print
    # Define a permutation tensor
    permutation = torch.tensor([2, 0, 1, 3], dtype=torch.int32)
    rev_perm = reverse_permutation(permutation)
    sparse_perm_matrix = permutation_to_sparse_csr(permutation, dtype=torch.float32)

    csr_mat = gen_rand_sp_matrix(4, 4, 0.4)
    c_print(csr_mat.to_dense(), color="bright_cyan")

    out = csr_mat.to_dense()[rev_perm]


    # Using CSRPermuter
    permuter = CSRPermuter(permutation, csr_mat)
    out_p = permuter.matrix_permute(csr_mat)

    print(out.to_dense())
    print(out_p.to_dense())

# Example usage:
if __name__ == "__main__":
    main()