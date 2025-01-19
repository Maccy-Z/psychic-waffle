import torch
tensor = torch.zeros(4, 4)
mask = torch.tensor([[True, True, False, False],
                     [False, False, True, True],
                     [True, False, False, False],
                     [False, True, True, False]])  # Mask with irregular `True`s
values = torch.tensor([1., 2, 3, 4, 5, 6, 7])  # Enough values for all `True`s

# Set the `True` positions
tensor[mask] = values
print(tensor)
print(tensor[mask].shape)