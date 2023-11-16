import torch
import gpytorch
from matplotlib import pyplot as plt
import math
from utils import c_print

class SimpleGPR(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SimpleGPR, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPR:
    model: SimpleGPR = None

    def __init__(self, noise_var=None, kern_var=None, kern_r=None, mean=None):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        if noise_var is not None:
            self.likelihood.noise = noise_var
            self.likelihood.noise_covar.raw_noise.requires_grad = False

    def fit(self, train_x, train_y, n_iters=None):
        self.model = SimpleGPR(train_x, train_y, self.likelihood)
        self.model.train()

        if n_iters is not None:
            self.fit_params(train_x, train_y, n_iters)

    def fit_params(self, train_x, train_y, n_iters):
        print("Fitting Params")
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

        for i in range(n_iters):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        # Print out final model parameters
        noise = self.model.likelihood.noise_covar.noise
        kern_r = self.model.covar_module.base_kernel.lengthscale
        kern_var = self.model.covar_module.outputscale

        c_print(f"Noise: {noise.squeeze().detach():.3g}", color="green")
        c_print(f"Kern R: {kern_r.squeeze().detach():.3g}", color="green")
        c_print(f"Kern Var: {kern_var.squeeze().detach():.3g}", color="green")

    def predict(self, test_x):
        self.model.eval()
        with torch.no_grad():
            pred_dist = self.model(test_x)
            pred_like = self.likelihood(pred_dist)

        out = {"mean": pred_dist.mean, "var": pred_dist.variance, "conf_region": pred_like.confidence_region()}

        return out


def main():
    model = GPR(noise_var=None)

    train_x = torch.linspace(0, 1, 100)
    train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.06)

    model.fit(train_x, train_y, n_iters=100)

    test_x = torch.linspace(0, 1, 51)
    out = model.predict(test_x)
    out = model.predict(test_x)

    test_mean = out['mean']
    lower, upper = out['conf_region']

    plt.scatter(train_x, train_y)
    plt.plot(test_x, test_mean)
    plt.fill_between(test_x, lower, upper, alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()
