from matplotlib import pyplot as plt
import numpy as np


def show_grid(u):
    """
    Visualize a 2D grid of values using matplotlib
    """
    # Check if the input is a 2D grid
    if u.ndim == 1:
        u = u.reshape(1, -1)
    elif u.ndim > 2:
        raise ValueError("Input must be a 1D or 2D array")
    print(u.ndim)
    plt.figure(figsize=(8, 6))
    plt.imshow(u, cmap='viridis', origin='lower')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    # Define the grid dimensions
    nx, ny = 10, 10

    # Create a 2D grid of u values (replace this with your actual u grid)
    u = np.random.rand(nx, ny)
    show_grid(u)
