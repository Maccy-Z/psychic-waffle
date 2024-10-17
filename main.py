import torch


class CSRRowMultiplier:
    def __init__(self, A_csr: torch.Tensor):
        """
        Initialize the CSRRowMultiplier with a CSR tensor.
        Args:
            A_csr (torch.Tensor): A sparse CSR tensor with fixed sparsity pattern.
        """
        if not A_csr.is_sparse_csr:
            raise ValueError("Input tensor must be a sparse CSR tensor.")

        self.A_csr = A_csr
        self.crow_indices = A_csr.crow_indices()
        self.col_indices = A_csr.col_indices()

        # Precompute row indices for each non-zero element
        row_lengths = self.crow_indices[1:] - self.crow_indices[:-1]
        self.row_indices = torch.arange(self.A_csr.size(0), device=self.A_csr.device).repeat_interleave(row_lengths)

    def multiply(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Multiply the CSR tensor A row-wise by vector b.
        """
        # Scale the values by the corresponding row elements
        values = A.values()
        scaled_values = values * b[self.row_indices]

        # Return a new CSR tensor with the scaled values
        return torch.sparse_csr_tensor(
            self.crow_indices,
            self.col_indices,
            scaled_values,
            size=self.A_csr.size(),
            device=self.A_csr.device
        )

# Example Usage
if __name__ == "__main__":
    # Example sparse CSR tensor A
    crow_indices = torch.tensor([0, 2, 3, 5], dtype=torch.int32)  # 3 rows
    col_indices = torch.tensor([0, 2, 1, 0, 2], dtype=torch.int32)
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    A_csr = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=(3, 3))

    # Initialize the multiplier with A_csr
    multiplier = CSRRowMultiplier(A_csr)

    # Vector b to multiply with
    b = torch.tensor([10.0, 20.0, 30.0])

    # Perform row-wise multiplication
    A_scaled = multiplier.multiply(A_csr, b)

    print("Original CSR Tensor:")
    print(A_csr.to_dense())
    print("\nScaled CSR Tensor:")
    print(A_scaled.to_dense())
