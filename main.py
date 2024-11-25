import numpy as np

A = np.array([[1, 2],
              [3, 4],
              [5, 6]], dtype=np.float64)

b = np.array([7, 8, 9], dtype=np.float64)

x = np.linalg.lstsq(A, b, rcond=None)[0]
print("Expected solution x:")
print(x)
