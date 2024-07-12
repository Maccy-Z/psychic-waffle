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

    plt.figure()
    plt.imshow(u.T.cpu().detach(), cmap='viridis', origin='lower')
    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def get_split_indices(tensor_size, m):
    """
    Get pairs of indices to split a 1D tensor into m chunks of possibly uneven size.

    Parameters:
    tensor_size (int): The size of the input 1D tensor.
    m (int): The number of chunks to split the tensor into.

    Returns:
    list of tuples: A list of (start, end) index pairs for each chunk.
    """
    split_sizes = [(tensor_size // m) + (1 if x < (tensor_size % m) else 0) for x in range(m)]

    indices = [0] + [sum(split_sizes[:i + 1]) for i in range(m)]

    index_pairs = [(indices[i], indices[i + 1]) for i in range(len(indices) - 1)]

    return index_pairs


if __name__ == "__main__":
    # Define the grid dimensions
    nx, ny = 10, 10

    # Create a 2D grid of u values (replace this with your actual u grid)
    u = np.random.rand(nx, ny)
    show_grid(u)
