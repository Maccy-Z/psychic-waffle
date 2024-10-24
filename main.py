import torch


class CSRConcatenator:
    def __init__(self, csr_tensor_A, csr_tensor_B):
        # Total number of rows and columns
        num_rows_A = csr_tensor_A.shape[0]
        num_rows_B = csr_tensor_B.shape[0]
        total_rows = num_rows_A + num_rows_B
        num_cols = csr_tensor_A.shape[1]

        # Precompute the output crow_indices
        self.output_crow_indices = torch.zeros(total_rows + 1, dtype=csr_tensor_A.crow_indices().dtype)
        self.output_crow_indices[1:num_rows_A + 1] = csr_tensor_A.crow_indices()[1:]
        total_nnz_A = csr_tensor_A.crow_indices()[-1]
        self.output_crow_indices[num_rows_A + 1:] = total_nnz_A + csr_tensor_B.crow_indices()[1:]

        # Precompute the output col_indices
        self.output_col_indices = torch.cat([csr_tensor_A.col_indices(), csr_tensor_B.col_indices()])

        # Store the output shape
        self.output_shape = (total_rows, num_cols)

    def concatenate(self, values_A, values_B):
        # Concatenate the values efficiently
        output_values = torch.cat([values_A, values_B])

        # Build and return the output CSR tensor
        output_csr = torch.sparse_csr_tensor(
            self.output_crow_indices,
            self.output_col_indices,
            output_values,
            size=self.output_shape
        )
        return output_csr



# Define the crow_indices and col_indices for tensors A and B
crow_indices_A = torch.tensor([0, 2, 4])  # Example for 2 rows
col_indices_A = torch.tensor([0, 2, 1, 2])
values_A = torch.tensor([1.0, 2.0, 3.0, 4.0])

crow_indices_B = torch.tensor([0, 2, 4])
col_indices_B = torch.tensor([0, 2, 1, 2])
values_B = torch.tensor([5.0, 6.0, 7.0, 8.0])

# Create CSR tensors A and B with the same sparsity pattern
csr_tensor_A = torch.sparse_csr_tensor(crow_indices_A, col_indices_A, values_A, size=(2, 3))
csr_tensor_B = torch.sparse_csr_tensor(crow_indices_B, col_indices_B, values_B, size=(2, 3))

# Initialize the concatenator with tensors A and B
concatenator = CSRConcatenator(csr_tensor_A, csr_tensor_B)

# Concatenate the values multiple times efficiently
output_csr = concatenator.concatenate(values_A, values_B)

# Output CSR tensor details
print("Output CSR tensor:")
print(csr_tensor_A.to_dense())
print(csr_tensor_B.to_dense())
print()
print(output_csr.to_dense())

