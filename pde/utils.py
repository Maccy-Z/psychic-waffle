from matplotlib import pyplot as plt
import numpy as np
import torch

def show_grid(u: torch.Tensor, title=None):
    """
    Visualize a 2D grid of values using matplotlib
    """
    # Check if the input is a 2D grid
    if u.ndim == 1:
        u = u.reshape(1, -1)
    elif u.ndim > 2:
        raise ValueError("Input must be a 1D or 2D array")

    plt.figure(figsize=(8, 6))
    plt.imshow(u.T.cpu().detach(), cmap='viridis', origin='lower')
    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.show()


if __name__ == "__main__":
    # Define the grid dimensions
    nx, ny = 10, 10

    # Create a 2D grid of u values (replace this with your actual u grid)
    u = np.random.rand(nx, ny)
    show_grid(u)
