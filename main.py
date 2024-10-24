import torch
import matplotlib.pyplot as plt
import numpy as np
from pde.utils_sparse import gen_rand_sp_matrix

# Example CSR data (small matrix for demonstration)
num_rows = 5
num_cols = 6
sparse_csr = gen_rand_sp_matrix(num_rows, num_cols, density=0.4, device='cpu')


def plot_sparsity(A):
    sparse_coo = A.to_sparse_coo().coalesce()
    indices = sparse_coo.indices()
    rows = indices[0].cpu().numpy()
    cols = indices[1].cpu().numpy()
    size = sparse_coo.size()

    dense_binary = np.zeros(size, dtype=np.int32)
    dense_binary[rows, cols] = 1

    # Plot using imshow
    plt.figure(figsize=(6, 5))
    plt.imshow(dense_binary, cmap='Greys', interpolation='none', aspect='auto')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.title('Sparsity Pattern')
    plt.gca().invert_yaxis()
    plt.show()

plot_sparsity(sparse_csr)