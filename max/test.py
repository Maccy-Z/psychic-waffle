import math

from matplotlib import pyplot as plt
import torch
import numpy as np

xs = torch.linspace(-0.1, 0.1, 21)
dys = \
    [0.128, 0.125, 0.122, 0.118, 0.112, 0.103, 0.0896, 0.0711, 0.0483, 0.0237, 0, -0.0208, -0.0378, -0.051, -0.0607, -0.0674, -0.0716, -0.0737, -0.0741, -0.0733, -0.0717, ]


def estimate_nth_derivative(xs, ys, n):
    """
    Estimates the nth derivative of a function given arrays of x and y values.

    Parameters:
    xs (array-like): Array of x-values.
    ys (array-like): Array of y-values corresponding to the x-values.
    n (int): The order of the derivative to be estimated.

    Returns:
    numpy.ndarray: Array of estimated nth derivatives at each x-value.
    """
    if n < 0:
        raise ValueError("n must be a non-negative integer")

    # Convert to numpy arrays for vectorized operations
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    for _ in range(n):
        # Calculate differences
        dx = np.diff(xs)
        dy = np.diff(ys)

        # Estimate derivatives
        derivatives = np.zeros_like(xs)
        derivatives[1:-1] = (dy[1:] / dx[1:] + dy[:-1] / dx[:-1]) / 2

        # Handle endpoints with one-sided differences
        derivatives[0] = dy[0] / dx[0]
        derivatives[-1] = dy[-1] / dx[-1]

        # Prepare for next iteration
        ys = derivatives

    return derivatives


deriv_ests = estimate_nth_derivative(xs, dys, 1)
d2_ests = estimate_nth_derivative(xs, dys, 2)
d3_ests = estimate_nth_derivative(xs, dys, 3)
print(deriv_ests[10])
print(d2_ests[10])
print(d3_ests[10])

c1 = -1.9984 - 0.2464
c2 = (2 * 9.1851e+00 + -6.0647e-01 + 12.4250)
c3 = -5.1439 * 1 / 6

print(f'{c1 = }, {c2 = }, {c3 = }')
cs = [c1, c2, ]

dys_pred = torch.zeros_like(torch.tensor(dys))
for i in range(2):
    dys_pred += cs[i] * xs ** (i + 1) / math.factorial(i + 1) # + c2 * xs ** 2 # + c3 * xs ** 3

    plt.title(f'{i + 1} order')
    plt.plot(xs, dys)
    plt.plot(xs, dys_pred)
    plt.show()
