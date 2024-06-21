import torch
from xitorch.optimize import rootfinder

class NNModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.A = torch.tensor([[1.1, 0.4], [0.3, 0.8]])
        self.b = torch.tensor([[0.3], [-0.2]])

    def forward(self, x):  # also called in __call__
        x = x[:2]
        print(x)
        fwd = torch.tanh(self.A @ x + self.b) + x / 2.0
        # print(fwd)
        return fwd


module = NNModule()
x0 = torch.zeros((3,1))  # zeros as the initial guess
xroot = rootfinder(module.forward, x0, params=())  # module.forward only takes x
print(xroot)
