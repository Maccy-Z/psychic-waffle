import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

from pde.utils_sparse import generate_random_sparse_matrix

np.set_printoptions(precision=2, suppress=True, linewidth=100)

# Define your sparse matrix (example)
A_sparse = generate_random_sparse_matrix(6, 6, 0.3, device="cpu")

# Convert to scipy
A_sparse = A_sparse.coalesce()
A = csr_matrix((A_sparse.values().numpy(), A_sparse.indices().numpy()), shape=A_sparse.size())

print(f'Original Matrix:\n{A.toarray()}')

# Apply Reverse Cuthill-McKee ordering
perm = reverse_cuthill_mckee(A)

print("Permutation order:", perm)

# Permute the matrix
A_permuted = A[perm, :][:, perm]

print("Permuted Matrix:\n", A_permuted.toarray())
