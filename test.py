# import torch
# import cupyx.scipy.sparse.linalg
#
# x = torch.randn((150**2, 150**2), device="cuda")

import torch
import cupy as cp
import cupyx.scipy.sparse as sp
import cupyx.scipy.sparse.linalg as sp_linalg

# Example sparse matrix A and vector b in PyTorch
A_torch = torch.tensor([[3, 0, 0], [0, 4, 0], [0, 0, 5]], dtype=torch.float32)
b_torch = torch.tensor([6, 8, 10], dtype=torch.float32)


# Convert PyTorch tensors to CuPy arrays
A_cupy = cp.asarray(A_torch)
b_cupy = cp.asarray(b_torch.cpu().numpy())

# Convert the dense matrix A_cupy to a sparse CSR matrix
A_sparse_cupy = sp.csr_matrix(A_cupy)

print(A_sparse_cupy)

# Solve the sparse linear system Ax = b using CuPy
x_cupy = sp_linalg.spsolve(A_sparse_cupy, b_cupy)

# Convert the solution back to a PyTorch tensor if needed
x_torch = torch.from_dlpack(x_cupy)

print("Solution x:", x_torch)