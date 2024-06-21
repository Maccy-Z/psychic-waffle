from gp import GPRSimple
import torch
from torchdiffeq import odeint
from matplotlib import pyplot as plt
from utils import KDTreeDict
import math


class GPModel(torch.nn.Module):
    def __init__(self, model, perturb_0=None):
        super().__init__()
        self.model = model
        self.forward_cache = KDTreeDict()
        self.x0 = None

        # Grid comb of perturbations
        self.n_perturb = 21
        self.grid = torch.linspace(-1, 1, self.n_perturb)
        perturb_mag = torch.zeros(self.n_perturb)
        if perturb_0 is not None:
            perturb_mag[5] = torch.tensor(perturb_0)
            perturb_mag[0] = torch.tensor(perturb_0)

        self.perturb_mag = torch.nn.Parameter(perturb_mag)

        # self.plot_perturbed()

    # Cache predictions to calculate sensitivity
    def forward(self, t, X, training=True):
        # print(X)
        # exit(2)
        x = X[..., 0].view(-1, 1)
        v = X[..., 1].view(-1, 1)

        F = self.model.predict(x)['mean'].view(-1, 1)

        if training:
            F = self.sum_of_gaussians(F, x)

        return torch.stack([v, F], dim=-1)

    def sum_of_gaussians(self, F, x):
        sigma = 1 / self.n_perturb
        for mean, mag in zip(self.grid, self.perturb_mag):
            norm_const = 1 / (sigma * math.sqrt(2 * math.pi))
            F = F + mag * torch.exp(-((x - mean) ** 2) / (2 * sigma ** 2)) * norm_const
        return F

    def plot_perturbed(self):
        xs = torch.linspace(-1, 1, 100)
        with torch.no_grad():
            ys_pert = self.forward(None, torch.stack([xs, torch.zeros_like(xs)], dim=1), training=True)[:, 0, 1]
            ys_normal = self.forward(None, torch.stack([xs, torch.zeros_like(xs)], dim=1), training=False)[:, 0, 1]
        plt.plot(xs, ys_pert)
        plt.plot(xs, ys_normal)
        plt.show()


def dxdt(t, X: torch.Tensor):
    x = X[..., 0]
    v = X[..., 1]
    F = - 0.5 * x - x ** 3
    return torch.stack([v, F], dim=-1)


def grid_constructor(f, y0, t):
    return torch.linspace(0, 6, 100)


def test_model(t_eval, perturb_0=None):
    x_train = torch.linspace(-1, 1, 5)
    v_train = torch.zeros_like(x_train)
    X_train = torch.stack([x_train, v_train], dim=1)

    # Fit model to dxdt
    dydx_train = dxdt(None, X_train)
    model = GPRSimple(noise_var=1e-4)
    model.fit(x_train.view(-1, 1), dydx_train[:, 1], n_iters=None)

    # Make predictions using model
    gp_model = GPModel(model, perturb_0=perturb_0)
    y_pred = odeint(gp_model.forward, torch.tensor([1., 0.]), t_eval, method='rk4',
                    #options={'grid_constructor': grid_constructor}
                    )

    if perturb_0 is None:
        # model.predict_and_plot(torch.linspace(-1, 1, 50))

        # Find gradient of objective w.r.t. F(t, x)
        # objective = y_pred[-1, 0]
        # objective.backward()
        #
        # grads = gp_model.perturb_mag.grad
        # print(grads)

        init_point = gp_model.perturb_mag

        dydf = torch.autograd.grad(y_pred[-1, 0], init_point, create_graph=True, retain_graph=True)[0]
        print(f'{dydf = }')

        d2ydf2 = torch.autograd.grad(dydf[0], init_point, create_graph=True, retain_graph=True)[0]
        print(f'{d2ydf2 = }')

        # d3ydf3 = torch.autograd.grad(d2ydf2[5], init_point)[0]
        # print(f'{d3ydf3 = }')


    # with torch.no_grad():
    #     plt.plot(xs_grad, grads)
    #     plt.show()
    #
    #     plt.plot(t_eval, y_pred)
    #     plt.show()
    #


    # xs, fs = [], []
    # dldf = []
    # for x, F in gp_model.forward_cache.items(sort_key="t"):
    #     xs.append(x), fs.append(F.item())
    #     dldf.append(F.grad.item())
    #
    # print(f'{dldf[:5] = }')
    # plt.plot(xs, fs)
    # plt.plot(xs, dldf)
    # plt.show()
    # print(gp_model.forward_cache.keys)
    # init_point = gp_model.forward_cache.values[1]
    # print(f'{init_point = }')
    #
    # dydf = torch.autograd.grad(y_pred[-1, 0], init_point, create_graph=True)[0]
    # print(f'{dydf = }')
    #
    # d2ydf2 = torch.autograd.grad(dydf, init_point)[0]
    # print(f'{d2ydf2 = }')
    # print()

    return y_pred, gp_model.forward_cache


def cached_model(t_eval, saved_F: KDTreeDict, eps=0.):
    def call_F(t, X):
        x = X[..., 0]
        v = X[..., 1]
        F = saved_F[x].squeeze()
        return torch.stack([v, F], dim=-1)

    saved_F.perturb(eps=eps)

    with torch.no_grad():
        y_cache_preds = odeint(call_F, torch.tensor([1., 0.]), t_eval, method='rk4',
                               options={'grid_constructor': grid_constructor})
    saved_F.perturb(eps=-eps)

    return y_cache_preds


def main():
    t_eval = torch.linspace(0, 6, 100)

    # True ODE
    # y_true = odeint(dxdt, torch.tensor([1., 0]), t_eval, method='rk4')

    # Predcted ODE
    y_pred, forward_cache = test_model(t_eval)

    dys = []
    for eps in torch.linspace(-0.1, 0.1, 21):
        with torch.no_grad():
            y_cache_preds = test_model(t_eval, perturb_0=eps)[0]

        dy = y_cache_preds[-1, 0] - y_pred[-1, 0]
        dy = dy.detach().item()
        dys.append(dy)

        # print(f'{eps = }, {dy = }, {y_cache_preds[-1, 0] = }')
    #
    print("Changes")
    print("[" + "".join([f'{dy:.3g}, ' for dy in dys]) + "]")
    #
    # with torch.no_grad():
    #     plt.plot(t_eval, y_pred, label="Preds")
    #     # plt.plot(t_eval, y_true, label="True")
    #     plt.plot(t_eval, y_cache_preds, label="Cached")
    #
    #     plt.legend()
    #     plt.show()


if __name__ == "__main__":
    main()
