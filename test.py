# import torch
#
# sparsity = torch.tensor([[ True,  True, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
#          False, False, False],
#         [ True,  True,  True, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
#          False, False, False],
#         [False,  True,  True,  True, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,
#          False, False, False],
#         [False, False,  True,  True, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False,
#          False, False, False],
#         [ True, False, False, False,  True,  True, False, False,  True, False, False, False, False, False, False, False, False, False, False, False,
#          False, False, False],
#         [False,  True, False, False,  True,  True,  True, False, False,  True, False, False, False, False, False, False, False, False, False, False,
#          False, False, False],
#         [False, False,  True, False, False,  True,  True,  True, False, False,  True, False, False, False, False, False, False, False, False, False,
#          False, False, False],
#         [False, False, False,  True, False, False,  True,  True, False, False, False,  True, False, False, False, False, False, False, False, False,
#          False, False, False],
#         [False, False, False, False,  True, False, False, False,  True,  True, False, False,  True, False, False, False, False, False, False, False,
#          False, False, False],
#         [False, False, False, False, False,  True, False, False,  True,  True,  True, False, False,  True, False, False, False, False, False, False,
#          False, False, False],
#         [False, False, False, False, False, False,  True, False, False,  True,  True,  True, False, False,  True, False, False, False, False, False,
#          False, False, False],
#         [False, False, False, False, False, False, False,  True, False, False,  True,  True, False, False, False,  True, False, False, False, False,
#          False, False, False],
#         [False, False, False, False, False, False, False, False,  True, False, False, False,  True,  True, False, False,  True, False, False, False,
#          False, False, False],
#         [False, False, False, False, False, False, False, False, False,  True, False, False,  True,  True,  True, False, False,  True, False, False,
#          False, False, False],
#         [False, False, False, False, False, False, False, False, False, False,  True, False, False,  True,  True,  True, False, False,  True, False,
#          False, False, False],
#         [False, False, False, False, False, False, False, False, False, False, False,  True, False, False,  True,  True, False, False, False,  True,
#          False, False, False],
#         [False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False,  True,  True, False, False,
#           True, False, False],
#         [False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False,  True,  True,  True, False,
#          False,  True, False],
#         [False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False,  True,  True,  True,
#          False, False,  True],
#         [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False,  True,  True,
#          False, False, False],
#         [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False,
#           True,  True, False],
#         [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False,
#           True,  True,  True],
#         [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False,
#          False,  True,  True]])
#
# eqs = [0, 1, 2, 3]
# wanted_cols = sparsity[eqs]
# nonzero_us = torch.nonzero(wanted_cols, as_tuple=True)[1]
# nonzero_us = torch.unique(nonzero_us)
#
# print(nonzero_us)
#
#
import torch


def create_band_matrix(diagonal_values, off_diagonal_values, n, size):
    """
    Creates a matrix with specified diagonal elements and elements n rows/columns above and below the diagonal.

    Parameters:
    diagonal_values (float): Value to be filled in the main diagonal.
    off_diagonal_values (float): Value to be filled in the diagonals n rows/columns above and below the main diagonal.
    n (int): Number of rows/columns above and below the main diagonal.
    size (int): Size of the square matrix.

    Returns:
    torch.Tensor: The resulting band matrix.
    """
    # Initialize the matrix with zeros
    matrix = torch.zeros(size, size)

    # Fill the main diagonal
    matrix += torch.diag(torch.full((size,), diagonal_values))

    # Fill the off diagonals
    for i in range(1, n + 1):
        matrix += torch.diag(torch.full((size - i,), off_diagonal_values), i)
        matrix += torch.diag(torch.full((size - i,), off_diagonal_values), -i)

    return matrix


# Example usage
diagonal_values = 1.0
off_diagonal_values = 0.5
n = 2
size = 5

band_matrix = create_band_matrix(diagonal_values, off_diagonal_values, n, size)
print(band_matrix)
