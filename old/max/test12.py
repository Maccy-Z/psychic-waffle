import torch
import math
from matplotlib import pyplot as plt
import numpy as np

means = torch.linspace(-0, 10, 11)
mags = torch.zeros_like(means)
x = torch.linspace(-0, 10, 100)
perturb_idx = [0, 1, 2, 3, 4, 5]
sigma = 0.5

for idx in perturb_idx:
    mags[idx] = 1

F = torch.zeros_like(x)
for mean, mag in zip(means, mags):
    norm_const = 0.85  # / (sigma * math.sqrt(2 * math.pi))
    F = F + mag * torch.exp(-((x - mean) ** 2) / (2 * sigma ** 2)) * norm_const

plt.plot(x, F)
# plt.plot(x, np.sin(x))
plt.show()
