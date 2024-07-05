import torch


x = torch.arange(9).view(3, 3)
print(x)

sl = (slice(1, 2), slice(1, 3))

print(x[sl])