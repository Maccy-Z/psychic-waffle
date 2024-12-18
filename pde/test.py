import torch

def test_sparse_csr_dense_multiplication():
    print("Starting")
    # Create a sparse CSR matrix
    indices = torch.tensor([[0, 0, 1], [0, 2, 1]])  # Coordinates of non-zero elements
    values = torch.tensor([1.0, 2.0, 3.0])  # Non-zero values
    size = (3, 3)  # Shape of the sparse matrix

    print("Making")
    sparse_csr_matrix = torch.sparse_csr_tensor(
        crow_indices=torch.tensor([0, 1, 3], dtype=torch.int32),
        col_indices=torch.tensor([0, 2, 1], dtype=torch.int32),
        values=values,
        size=size
    ).cuda()

    print("Dense")
    # Create a dense matrix
    dense_matrix = torch.tensor([
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [1.0, 2.0, 3.0]
    ]).cuda()

    print("MM")
    # Perform sparse CSR x dense multiplication
    result = torch.mm(sparse_csr_matrix, dense_matrix)

    print("Sparse CSR Matrix:")
    print(sparse_csr_matrix)
    print("\nDense Matrix:")
    print(dense_matrix)
    print("\nResult of Sparse CSR x Dense Multiplication:")
    print(result)

# Call the test function
test_sparse_csr_dense_multiplication()

