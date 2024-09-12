
import torch
from utils import show_grid


dfdus = torch.load("dfdu.pth", weights_only=True)

dfdu, dfduX, dfduXX = dfdus
h = 1


# Start with a single row
dfdu, dfduX, dfduXX = dfdu[0, :, 1], dfduX[0, :, 1], dfduXX[0, :, 1]
N = dfdu.shape[0]

transform = torch.tensor([[1, 0, -1/h**2], [0, 1/(2*h), 1/h**2], [0, -1/(2*h), 1/h**2]], device='cuda')
f_abc = torch.stack([dfdu, dfduX, dfduXX])
print(f_abc.shape)
f_us = torch.matmul(transform, f_abc)
print(f_us.shape)
#
# jacobian = torch.zeros((N, N))
#
# # Diagonal elements
# jacobian.diagonal().copy_(dfdu - 2 / h**2 * dfduXX)
# # Offset terms
# print(dfduXX[0])
# jacobian.diagonal(offset=-1).copy_(1 / h**2 * dfduXX[:-1])
# jacobian.diagonal(offset=1).copy_(1 / h**2 * dfduXX[1:])
