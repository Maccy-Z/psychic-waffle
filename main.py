import torch
from sparse_tensor import gen_rand_sp_matrix


def select_rows_coo(sparse_coo: torch.sparse_coo_tensor, row_mask: torch.Tensor) -> torch.sparse_coo_tensor:
    """
    Selects rows from a COO sparse tensor based on a row-wise mask.
    Args:
        sparse_coo (torch.sparse_coo_tensor): The input sparse COO tensor.
        row_mask (torch.Tensor): A 1D boolean tensor where True indicates the row to keep.
    Returns:
        torch.sparse_coo_tensor: A new sparse COO tensor with only the selected rows.
    """
    # Extract indices and values from the sparse tensor
    indices = sparse_coo.coalesce().indices()  # Shape: [ndim, nnz]
    values = sparse_coo.coalesce().values()  # Shape: [nnz]

    # Assume the first dimension corresponds to rows
    row_indices = indices[0]

    # Create a mask for non-zero elements in the selected rows
    mask = row_mask[row_indices]

    # Apply the mask to filter indices and values
    selected_indices = indices[:, mask]
    selected_values = values[mask]

    # Get the selected row numbers in sorted order
    selected_rows = row_mask.nonzero(as_tuple=False).squeeze()

    # Ensure selected_rows is 1D
    if selected_rows.dim() == 0:
        selected_rows = selected_rows.unsqueeze(0)

    # Create a mapping from old row indices to new row indices
    # This ensures that the new tensor has contiguous row indices starting from 0
    # Example: If rows 1 and 3 are selected, row 1 -> 0 and row 3 -> 1 in the new tensor
    row_mapping = torch.arange(len(selected_rows), device=selected_rows.device)
    # Create a dictionary-like mapping using scatter
    mapping = torch.full((sparse_coo.size(0),), -1, dtype=torch.long, device=selected_rows.device)
    mapping[selected_rows] = row_mapping
    # Map the selected row indices
    new_row_indices = mapping[selected_indices[0]]

    if (new_row_indices == -1).any():
        raise RuntimeError("Some row indices were not mapped correctly.")

    # Replace the row indices with the new row indices
    new_indices = selected_indices.clone()
    new_indices[0] = new_row_indices

    # Define the new size: number of selected rows and the remaining dimensions
    new_size = [row_mask.sum().item()] + list(sparse_coo.size())[1:]

    # Create the new sparse COO tensor
    new_sparse_coo = torch.sparse_coo_tensor(new_indices, selected_values, size=new_size)

    return new_sparse_coo

def select_columns_coo(sparse_coo: torch.sparse_coo_tensor, col_mask: torch.Tensor) -> torch.sparse_coo_tensor:
    """
    Selects columns from a COO sparse tensor based on a column-wise mask.

    Args:
        sparse_coo (torch.sparse_coo_tensor): The input sparse COO tensor.
        col_mask (torch.Tensor): A 1D boolean tensor where True indicates the column to keep.

    Returns:
        torch.sparse_coo_tensor: A new sparse COO tensor with only the selected columns.
    """
    # Extract indices and values from the sparse tensor
    sparse_coo = sparse_coo.coalesce()  # Ensure indices are coalesced
    indices = sparse_coo.indices()      # Shape: [ndim, nnz]
    values = sparse_coo.values()        # Shape: [nnz]

    # Assume the second dimension corresponds to columns
    col_indices = indices[1]

    # Create a mask for non-zero elements in the selected columns
    mask = col_mask[col_indices]

    # Apply the mask to filter indices and values
    selected_indices = indices[:, mask]
    selected_values = values[mask]

    # Get the selected column numbers in sorted order
    selected_cols = col_mask.nonzero(as_tuple=False).squeeze()

    # Ensure selected_cols is 1D
    if selected_cols.dim() == 0:
        selected_cols = selected_cols.unsqueeze(0)

    # Create a mapping from old column indices to new column indices
    # This ensures that the new tensor has contiguous column indices starting from 0
    row_mapping = torch.arange(len(selected_cols), device=selected_cols.device)
    # Initialize a mapping tensor with -1 (invalid)
    mapping = torch.full((sparse_coo.size(1),), -1, dtype=torch.long, device=selected_cols.device)
    # Assign new indices to the selected columns
    mapping[selected_cols] = row_mapping
    # Map the selected column indices
    new_col_indices = mapping[selected_indices[1]]

    if (new_col_indices == -1).any():
        raise RuntimeError("Some column indices were not mapped correctly.")

    # Replace the column indices with the new column indices
    new_indices = selected_indices.clone()
    new_indices[1] = new_col_indices

    # Define the new size: number of rows remains the same, number of selected columns
    new_size = list(sparse_coo.size())
    new_size[1] = col_mask.sum().item()

    # Create the new sparse COO tensor
    new_sparse_coo = torch.sparse_coo_tensor(new_indices, selected_values, size=new_size)

    return new_sparse_coo


A = gen_rand_sp_matrix(5, 5, 0.5).coalesce()
B = torch.tensor([0, 0, 1, 1, 0], dtype=torch.bool)

C = select_columns_coo(A, B)

print(A.to_dense())
print(f'{C.to_dense()}')




