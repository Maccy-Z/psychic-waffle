import torch
import torch.nn as nn

model = torch.load("model.pt", map_location='cpu')

# Extract weights and biases
W1 = model.lin1.weight.data
b1 = model.lin1.bias.data
W2 = model.lin2.weight.data
b2 = model.lin2.bias.data

# Compute combined weights and biases
W_combined = torch.matmul(W2, W1)
b_combined = torch.matmul(W2, b1) + b2

# Create the combined linear layer
combined_linear = nn.Linear(5, 1, bias=True)
combined_linear.weight.data = W_combined
combined_linear.bias.data = b_combined

print(W_combined)
print(b_combined)