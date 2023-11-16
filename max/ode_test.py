from matplotlib import pyplot as plt
from torchdiffeq import odeint
import torch
import torch.nn as nn


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, 1)
        # torch.nn.init.zeros_(self.l3.bias)
        # torch.nn.init.zeros_(self.l3.weight)

        self.act = nn.functional.elu

        self.opt = torch.optim.Adam(self.parameters(), lr=1e-2)

    def forward(self, x, t=None):
        if x.dim() == 0:
            x = x.reshape(1)
        x = self.l1(x)
        x = self.act(self.l2(x))
        x = self.l3(x)

        return x

    def train_fn(self, x_train, y_train):
        losses = []
        for _ in range(500):
            y_pred = odeint(self, torch.tensor([0.]), x_train, method='rk4')
            loss = torch.mean((y_pred - y_train) ** 2 + torch.abs(y_pred - y_train))

            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

            losses.append(loss.detach())

        return losses

    def pred_ode(self, x_test):
        return odeint(self, torch.tensor([0.]), x_test)


def dydx_fn(x, t):
    return torch.cos(x)


def y_fn(x):
    y = odeint(dydx_fn, torch.tensor([0.]), x)
    return y


def main():
    # Train model
    x_train = torch.tensor([1., 2., 4., 5.])
    y_train = y_fn(x_train)

    x_test = torch.linspace(1, 5, 20)
    y_test = y_fn(x_test)

    model = Model()

    losses = model.train_fn(x_train, y_train)

    with torch.no_grad():
        y_pred = model.pred_ode(x_test)

    plt.scatter(x_train, y_train)
    plt.plot(x_test, y_test)
    plt.plot(x_test, y_pred)
    plt.show()

    plt.plot(losses)
    plt.ylim([0, 0.5])
    plt.show()


if __name__ == "__main__":
    main()
