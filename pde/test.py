import torch
from utils_sparse import plot_sparsity


A = torch.load("jacobian.pt", weights_only=True)
b = torch.load("residuals.pt", weights_only=True)
plot_sparsity(A)

A = A.to_dense()
b = b.to_dense()
Ab = torch.hstack([A, b.unsqueeze(1)])

print(A.shape)

print(torch.linalg.matrix_rank(A))
print(torch.linalg.matrix_rank(Ab))

x = torch.linalg.solve(A, b)
error = torch.max(torch.abs(A @ x - b))
print(error)
# print(x)

x2 = torch.linalg.lstsq(A, b).solution
error = torch.max(torch.abs(A @ x2 - b))
print(f'{error = }')