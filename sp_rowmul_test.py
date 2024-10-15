from sparse_tensor import gen_rand_sp_matrix
import torch

class SparseTensorSummer:
    """ Sum together multiple sparse CSR tensors with the same sparsity pattern. """
    def __init__(self, B_list):
        """
        Precompute the output CSR structure and mappings for efficient summation.
        Parameters:
        - B_list: List of K initial sparse CSR tensors (torch.sparse_csr_tensor).
        """
        self.size = B_list[0].size()
        self.device = B_list[0].device
        self.dtype = B_list[0].dtype

        # Store the initial crow_indices and col_indices of each B_k
        self.initial_crow_indices_list = [B.crow_indices() for B in B_list]
        self.initial_col_indices_list = [B.col_indices() for B in B_list]

        # Precompute the CSR structure and mappings
        self.output_crow_indices, self.output_col_indices, self.index_mapping_list = self.precompute_output_csr_structure(B_list)

    def precompute_output_csr_structure(self, B_list):
        """
        Precompute the CSR structure (crow_indices, col_indices) of the output tensor
        and the mappings from input tensors to output positions.

        Parameters:
        - B_list: List of K sparse CSR tensors.

        Returns:
        - output_crow_indices: crow_indices for the output CSR tensor.
        - output_col_indices: col_indices for the output CSR tensor.
        - index_mapping_list: List of mappings from each B_k's values to output values.
        """
        # Collect all non-zero indices from B_list
        all_row_indices = []
        all_col_indices = []
        nnz_per_tensor = []

        for B in B_list:
            crow_indices = B.crow_indices()
            col_indices = B.col_indices()
            num_rows = crow_indices.size(0) - 1
            row_indices = torch.repeat_interleave(
                torch.arange(num_rows, device=self.device),
                crow_indices[1:] - crow_indices[:-1]
            )
            all_row_indices.append(row_indices)
            all_col_indices.append(col_indices)
            nnz_per_tensor.append(col_indices.size(0))

        # Stack and get unique indices
        all_indices = torch.cat(
            [torch.stack([r, c], dim=1) for r, c in zip(all_row_indices, all_col_indices)],
            dim=0
        )
        unique_indices, inverse_indices = torch.unique(
            all_indices, dim=0, return_inverse=True
        )

        # Sort the unique indices to build CSR structure
        num_cols = self.size[1]
        sorted_order = torch.argsort(unique_indices[:, 0] * num_cols + unique_indices[:, 1])
        sorted_unique_indices = unique_indices[sorted_order]
        row_indices = sorted_unique_indices[:, 0]
        col_indices = sorted_unique_indices[:, 1]

        # Build output_crow_indices
        num_rows = self.size[0]
        row_counts = torch.bincount(row_indices, minlength=num_rows)
        output_crow_indices = torch.zeros(num_rows + 1, dtype=torch.long, device=self.device)
        output_crow_indices[1:] = torch.cumsum(row_counts, dim=0)
        output_col_indices = col_indices

        # Map each input tensor's indices to the output positions
        num_unique = unique_indices.size(0)
        unique_indices_to_sorted = torch.empty(
            num_unique, dtype=torch.long, device=self.device
        )
        unique_indices_to_sorted[sorted_order] = torch.arange(
            num_unique, device=self.device
        )

        cumulative_nnz = [0] + list(
            torch.cumsum(torch.tensor(nnz_per_tensor, device=self.device), dim=0).cpu().numpy()
        )
        index_mapping_list = []

        for k in range(len(B_list)):
            start = cumulative_nnz[k]
            end = cumulative_nnz[k+1]
            indices_k = inverse_indices[start:end]
            positions_k = unique_indices_to_sorted[indices_k]
            index_mapping_list.append(positions_k)

        return output_crow_indices, output_col_indices, index_mapping_list

    def sum_tensors(self, B_list_new) -> torch.sparse_csr_tensor:
        """
        Sum the values from B_list_new into the output CSR tensor using precomputed mappings.

        Parameters:
        - B_list_new: List of K sparse CSR tensors with new values but same sparsity patterns.

        Returns:
        - J: The output sparse CSR tensor representing J_{ij} = sum_k B_{ijk}.
        """
        # Check that the number of tensors matches
        assert len(B_list_new) == len(self.index_mapping_list), (
            "The number of tensors in B_list_new must match the initial B_list."
        )

        for k, B in enumerate(B_list_new):
            # Ensure the sparsity pattern matches the initial B_k
            if not torch.equal(B.crow_indices(), self.initial_crow_indices_list[k]):
                raise ValueError(f"Sparsity pattern of B_list_new[{k}] does not match the initial B_list.")
            if not torch.equal(B.col_indices(), self.initial_col_indices_list[k]):
                raise ValueError(f"Sparsity pattern of B_list_new[{k}] does not match the initial B_list.")

        # Initialize the output values tensor
        nnz_total = self.output_col_indices.size(0)
        output_values = torch.zeros(nnz_total, dtype=self.dtype, device=self.device)

        for k, B in enumerate(B_list_new):
            B_values = B.values()
            positions = self.index_mapping_list[k]
            output_values.index_add_(0, positions, B_values)

        # Create the output CSR tensor
        J = torch.sparse_csr_tensor(
            self.output_crow_indices, self.output_col_indices, output_values, size=self.size
        )
        return J


# Sample sparse CSR tensors with the same sparsity patterns

# Tensor B1
B1 = gen_rand_sp_matrix(4, 4, 0.2).to_sparse_csr()
B2 = gen_rand_sp_matrix(4, 4, 0.2).to_sparse_csr()
B3 = gen_rand_sp_matrix(4, 4, 0.2).to_sparse_csr()

# Initialize the SparseTensorSummer with the initial B_list
B_list_initial = [B1, B2, B3]
summer = SparseTensorSummer(B_list_initial)


C = summer.sum_tensors(B_list_initial)

C_d = C.to_dense()
c_true = B1.to_dense() + B2.to_dense() + B3.to_dense()

assert torch.allclose(C_d, c_true)

# # Now, suppose we have new values for B1 and B2 in multiple iterations
# for iteration in range(3):
#     # Generate new values (for demonstration purposes, multiply by iteration index)
#     values1_new = values1 * (iteration + 1)
#     values2_new = values2 * (iteration + 1)
#
#     # Create new B1 and B2 with updated values (sparsity patterns remain the same)
#     B1_new = torch.sparse_csr_tensor(crow_indices1, col_indices1, values1_new, size=(2, 4))
#     B2_new = torch.sparse_csr_tensor(crow_indices2, col_indices2, values2_new, size=(2, 4))
#     B_list_new = [B1_new, B2_new]
#
#     # Use the sum_tensors method to compute the sum
#     J = summer.sum_tensors(B_list_new)
#
#     # Convert to dense for visualization
#     print(f"Iteration {iteration + 1}:\n", J.to_dense())
