import numpy as np


def construct_A_numpy(N, h):
    A = np.zeros((N, N))

    # Left boundary (forward difference)
    A[0, 0] = -1.0 / h
    A[0, 1] = 1.0 / h

    # Interior points (central difference)
    for i in range(1, N - 1):
        A[i, i - 1] = -0.5 / h
        A[i, i + 1] = 0.5 / h

    # Right boundary (backward difference)
    A[N - 1, N - 2] = -1.0 / h
    A[N - 1, N - 1] = 1.0 / h

    return A

A = construct_A_numpy(5, 1.0)

print(A)

D2 = A @ A
print(D2)
