import torch

# Step 1: Define the original Jacobian matrix J
# For demonstration, let's create a 4x4 Jacobian matrix with distinct values
J = torch.tensor([
    [1.0,  2.0,  3.0,  4.0],
    [5.0,  6.0,  7.0,  8.0],
    [9.0, 10.0, 11.0, 12.0],
    [13.0,14.0, 15.0, 16.0]
])

print("Original Jacobian J:")
print(J)

# Step 2: Define new orderings for equations (rows) and variables (columns)
# Let's say we want to reorder the equations and variables to make J as diagonal as possible
# New orderings (permutations)
p1 = torch.tensor([2, 0, 3, 1], dtype=torch.long)  # New row order
p2 = torch.tensor([0, 1, 2, 3], dtype=torch.long)  # New column order

print("\nPermutation indices for rows (p1):", p1)
print("Permutation indices for columns (p2):", p2)

# Step 3: Permute the Jacobian matrix using advanced indexing
J_new = J[p1, :][:, p2]

print("\nPermuted Jacobian J_new:")
print(J_new)


# Step 4: Verify the permutation using permutation matrices (optional)
# Note: This step is optional and less efficient for large matrices

# Create permutation matrices
P1 = torch.eye(J.size(0))[p1]  # Permutation matrix for rows
P2 = torch.eye(J.size(1))[:, p2]  # Permutation matrix for columns

# Compute J_new_alt using matrix multiplication
J_new_alt = P1 @ J @ P2

print("\nPermuted Jacobian J_new_alt using permutation matrices:")
print(J_new_alt)

# Verify that both methods give the same result
assert torch.allclose(J_new, J_new_alt), "The permuted matrices are not equal!"

print("\nVerification passed: J_new and J_new_alt are equal.")
