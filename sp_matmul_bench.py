import torch
import torch_sparse
import time
from pde.utils_sparse import gen_rand_sp_matrix


def time_fn(fn, A, b):
    for _ in range(30):
        fn(A, b)
    torch.cuda.synchronize()
    st = time.time()
    for _ in range(100):
        fn(A, b)
    torch.cuda.synchronize()
    print(f'{fn.__name__} time = {time.time() - st}')


def loss_fn(c):
    # L = c.sum()
    # L.backward()
    pass
    return

def torch_mv(A, b):
    b.grad = None
    c = torch.mv(A, b)
    loss_fn(c)

    return c


def tsp_spmm(A, b):
    b.grad = None
    r, c, vals = A.coo()
    index = (r, c)
    c = torch_sparse.spmm(index, vals, rows, cols, b.unsqueeze(1)).squeeze()
    loss_fn(c)
    return c


def tsp_mm(A, b):
    b.grad = None
    c = torch_sparse.matmul(A, b.unsqueeze(1)).squeeze()
    loss_fn(c)
    return c


rows, cols = 100_000, 100_000
density = 0.001

A = gen_rand_sp_matrix(rows, cols, density, device="cuda")#.to_sparse_coo()

crow, col, val = A.crow_indices(), A.col_indices(), A.values()
size = A.size()

crow, col = crow.to(torch.int32), col.to(torch.int32)
A = torch.sparse_csr_tensor(crow, col, val, size, device="cuda")

b = torch.randn(cols, requires_grad=True, device="cuda")

time_fn(torch_mv, A, b)

# A_torchsp = torch_sparse.SparseTensor.from_torch_sparse_coo_tensor(A)
#
# time_fn(tsp_spmm, A_torchsp, b)
# time_fn(tsp_mm, A_torchsp, b)
#
# tsp_spmm = tsp_spmm(A_torchsp, b)
# tsp_mm = tsp_mm(A_torchsp, b)
#
# assert torch.allclose(tsp_spmm, torch_mv, atol=1e-4)
# assert torch.allclose(tsp_spmm, tsp_mm, atol=1e-4)

