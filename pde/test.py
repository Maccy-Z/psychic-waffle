
import torch
from utils import show_grid


dfdus = torch.load("dfdu.pth", weights_only=True)

dfdu, dfduX, dfduXX = dfdus
h = 1


# Start with a single row
dfdu, dfduX, dfduXX = dfdu[0, :, 1], dfduX[0, :, 1], dfduXX[0, :, 1]

N = dfdu.shape[0]
jacobian = torch.zeros((N, N))

# Diagonal elements
jacobian.diagonal().copy_(dfdu - 2 / h**2 * dfduXX)

# Offset terms
print(dfduXX[0])
jacobian.diagonal(offset=-1).copy_(1 / h**2 * dfduXX[:-1])
jacobian.diagonal(offset=1).copy_(1 / h**2 * dfduXX[1:])

# jacobian.diagonal(offset=-1).copy_(dfduX[0][1:-1])

show_grid(jacobian, origin="upper")
