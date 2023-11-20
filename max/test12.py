import torch
from matplotlib import pyplot as plt
import math

x = torch.arange(-5, 5, 0.1)
sigma = 1

norm_const = 1 / (sigma * math.sqrt(2 * math.pi))
y = torch.exp(-((x) ** 2) / (2 * sigma ** 2)) * norm_const
plt.plot(x, y)
plt.show()