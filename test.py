from torchsparsegradutils import sparse_solve
from torchsparsegradutils.utils import linear_cg
import torch


#A = torch.diag(torch.randn(10))
A = torch.randn([10, 10])
b = torch.randn([10])


x = sparse_solve.sparse_generic_solve(A, b)

b_pred = A @ x

print(b)
print(b_pred)



