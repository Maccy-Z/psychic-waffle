import torch

class SparseCSRTensorBuilder:
    def __init__(self, total_rows, total_cols, device=None):
        """
        Initializes the builder for a CSR sparse tensor.
        Parameters:
        - total_rows: int, total number of rows in the matrix.
        - total_cols: int, total number of columns in the matrix.
        - device: torch device (optional).
        """
        self.total_rows = total_rows
        self.total_cols = total_cols
        self.device = device

        # Internal storage for CSR components
        self.nnz_per_row = torch.tensor([0] * total_rows, device=device, dtype=torch.int32)  # Number of non-zero elements per row
        self.col_indices = []                # Column indices of non-zero elements
        self.values = []                     # Non-zero values

    def add_block(self, block_dense_values, block_row_offset, block_col_offset):
        """
        Adds a dense block to the CSR components using efficient tensor operations.

        Parameters:
        - block_dense_values: 2D tensor (n x m), dense block of values.
        - block_row_offset: int, the starting row index of the block in the overall matrix.
        - block_col_offset: int, the starting column index of the block in the overall matrix.
        """
        n, m = block_dense_values.shape

        block_sparse = block_dense_values.to_sparse_csr()

        # Count non-zero elements per row in the block
        crow_idxs = block_sparse.crow_indices()
        counts = crow_idxs[1:] - crow_idxs[:-1]

        # Update nnz_per_row for the corresponding global rows
        # Using scatter_add for efficient batch updates
        self.nnz_per_row[block_row_offset:block_row_offset + n] += counts

        # Calculate global column indices
        global_cols = block_col_offset + block_sparse.col_indices()
        self.col_indices.append(global_cols)

        # Extract the non-zero values
        non_zero_values = block_sparse.values()
        self.values.append(non_zero_values)


    def build(self):
        """
        Builds and returns the sparse CSR tensor from the accumulated components.

        Returns:
        - csr_tensor: torch.sparse_csr_tensor, the constructed sparse CSR tensor.
        """
        # Compute crow_indices by cumulatively summing nnz_per_row
        crow_indices = torch.cat([
            torch.tensor([0], dtype=torch.int32, device=self.device),
            torch.cumsum(self.nnz_per_row, dim=0, dtype=torch.int32)
        ])

        # Convert col_indices and values to tensors
        col_indices_tensor = torch.cat(self.col_indices).to(torch.int32)
        values_tensor = torch.cat(self.values)

        # Create the sparse CSR tensor
        csr_tensor = torch.sparse_csr_tensor(
            crow_indices,
            col_indices_tensor,
            values_tensor,
            size=(self.total_rows, self.total_cols),
        )

        return csr_tensor


def test_matrix(rows, cols, row_block):
    x = torch.tensor([[1., 0,], [1, 1], [0, 1]]) * (row_block + 1)
    col_block = row_block

    return x, row_block, col_block


# Example usage:
if __name__ == "__main__":
    # Initialize the builder with total dimensions
    total_rows = 10  # Total number of rows in the matrix
    total_cols = 10  # Total number of columns in the matrix
    builder = SparseCSRTensorBuilder(total_rows, total_cols, device="cpu")

    for i in range(3):
        block_vals, row_block, col_block = test_matrix(3, 2, i * 3)

        print(f'{row_block = }, {col_block = }')
        if i == 1:
            block_vals = torch.zeros_like(block_vals)
        builder.add_block(block_vals, row_block, col_block)


    # builder.add_block(block_dense_values, block_row_offset, block_col_offset)

    # Build the CSR tensor
    csr_tensor = builder.build()

    # csr_tensor now contains the sparse matrix constructed from the blocks
    print("CSR crow_indices:", csr_tensor.crow_indices())
    print("CSR col_indices:", csr_tensor.col_indices())
    print("CSR values:", csr_tensor.values())

    # Convert to dense tensor to visualize
    dense_tensor = csr_tensor.to_dense()
    print("Dense representation:\n", dense_tensor)

