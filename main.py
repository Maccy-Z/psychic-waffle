import torch
import matplotlib.pyplot as plt
import numpy as np
from pde.utils_sparse import gen_rand_sp_matrix

# Example CSR data (small matrix for demonstration)
num_rows = 5
num_cols = 6
sparse_csr = gen_rand_sp_matrix(num_rows, num_cols, density=0.4, device='cpu')


def CSRToInt32(sparse_csr: torch.Tensor) -> torch.Tensor:
    """
    Converts the crow_indices and col_indices of a sparse CSR tensor from int64 to int32.

    Args:
        sparse_csr (torch.Tensor): A PyTorch sparse CSR tensor.

    Returns:
        torch.Tensor: A new sparse CSR tensor with int32 indices and the same values.
    """
    # Extract CSR components
    crow_indices = sparse_csr.crow_indices()
    col_indices = sparse_csr.col_indices()
    values = sparse_csr.values()
    size = sparse_csr.size()
    dtype = values.dtype
    device = sparse_csr.device

    # Convert indices to int32
    crow_indices_int32 = crow_indices.to(torch.int32)
    col_indices_int32 = col_indices.to(torch.int32)

    # Reconstruct the sparse CSR tensor with int32 indices
    sparse_csr_int32 = torch.sparse_csr_tensor(
        crow_indices_int32,
        col_indices_int32,
        values,
        size=size,
        dtype=dtype,
        device=device
    )

    return sparse_csr_int32

sparse_csr_32 = convert_csr_indices_to_int32(sparse_csr)

print(sparse_csr_32.crow_indices())
