from matplotlib import pyplot as plt
import numpy as np
import torch


def show_grid(u: torch.Tensor, title=None, origin="lower"):
    """
    Visualize a 2D grid of values using matplotlib
    """
    # Check if the input is a 2D grid
    if u.ndim == 1:
        u = u.reshape(1, -1)
    elif u.ndim > 2:
        raise ValueError("Input must be a 1D or 2D array")

    plt.figure()
    u = u.T.cpu().detach().numpy()

    plt.imshow(u, cmap='viridis', origin=origin)
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


def setup_logging():
    import logging
    import sys
    import matplotlib

    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='\033[31m%(levelname)s: \033[33m%(message)s \033[0m')
