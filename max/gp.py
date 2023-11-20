import torch
import gpytorch
from matplotlib import pyplot as plt
import math
from utils import c_print, ParamHolder


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
    train_x = None
    train_y = None

    def __init__(self, noise_var=None, kern_var=None, kern_r=None, mean=None):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
        if noise_var is not None:
            self.likelihood.noise = noise_var
            self.likelihood.noise_covar.raw_noise.requires_grad = False

    def fit(self, train_x, train_y, n_iters=None):
        self.train_x, self.train_y = train_x, train_y

        self.model = SimpleGPR(train_x, train_y, self.likelihood)
        self.model.train()

        if n_iters is not None:
            self.fit_params(n_iters)

    def fit_params(self, n_iters):
        print("Fitting Params")
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

        for i in range(n_iters):
            optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            optimizer.step()

        print(f"Final Loss: {loss.item():.5g}")

        # Print out final model parameters
        noise = self.model.likelihood.noise_covar.noise
        kern_r = self.model.covar_module.base_kernel.lengthscale
        kern_scale = self.model.covar_module.outputscale

        c_print(f"Noise: {noise.squeeze().detach():.3g}", color="green")
        c_print(f"Kern R: {kern_r.squeeze().detach():.3g}", color="green")
        c_print(f"Kern var: {kern_scale.squeeze().detach():.3g}", color="green")

    def predict(self, test_x):
        self.model.eval()
        with torch.no_grad():
            pred_dist = self.model(test_x)
            pred_like = self.likelihood(pred_dist)

        out = {"mean": pred_dist.mean, "var": pred_dist.variance, "conf_region": pred_like.confidence_region(), "cov_matrix": pred_dist.covariance_matrix}
        return out

    def predict_and_plot(self, test_x):
        preds = self.predict(test_x)

        test_mean = preds['mean']
        lower, upper = preds['conf_region']

        plt.scatter(self.train_x, self.train_y)
        plt.plot(test_x, test_mean)
        plt.fill_between(test_x.squeeze(), lower, upper, alpha=0.5)
        plt.show()

    def predict_mean(self, test_x):
        preds = self.predict(test_x)
        return preds['mean']


class GPRSimple(torch.nn.Module):
    def __init__(self, noise_var=None, kern_len=math.log(2), kern_scale=math.log(2)):
        super(GPRSimple, self).__init__()
        self.param_holder = ParamHolder(noise_var=noise_var, kern_len=kern_len, kern_scale=kern_scale)

        self.is_fitted = False

    def kernel(self, X1, X2):
        """
        X1.shape = (n, dim)
        X2.shape = (m, dim)
        """
        X1 = X1.unsqueeze(1)  # Shape: [n, 1, d]
        X2 = X2.unsqueeze(0)  # Shape: [1, m, d]

        kern_scale, kern_len, _ = self.param_holder.get_params()

        sqdist = torch.norm(X1 - X2, dim=2, p=2).pow(2)
        return kern_scale * torch.exp(-0.5 * sqdist / kern_len ** 2)

    def fit(self, train_x: torch.Tensor, train_y: torch.Tensor, n_iters=None):
        assert train_x.dim() == 2, "train_x must be a 2D tensor."
        self.train_x, self.train_y = train_x, train_y

        if n_iters is not None:
            self.fit_params(n_iters)

        _, _, noise = self.param_holder.get_params()

        # Cache important matrices
        K = self.kernel(train_x, train_x) + noise * torch.eye(len(train_x))
        self.L = torch.linalg.cholesky(K)
        self.inv_y = torch.cholesky_solve(train_y.unsqueeze(1), self.L)

        self.is_fitted = True

    # Log likelihood for fitting params
    def log_marginal_likelihood(self):
        _, _, noise = self.param_holder.get_params()

        K = self.kernel(self.train_x, self.train_x) + noise * torch.eye(len(self.train_x))
        L = torch.linalg.cholesky(K)
        inv_y = torch.cholesky_solve(self.train_y.unsqueeze(1), L)

        # Negative log-likelihood
        inv_quad = self.train_y.view(1, -1) @ inv_y
        logdet = torch.logdet(L)

        log_likelihood = 0.5 * (inv_quad + 2 * logdet + len(self.train_x) * torch.log(torch.tensor(2 * torch.pi)))

        # print(f'{inv_quad=}, {logdet=}')
        # print(torch.logdet(L))
        return log_likelihood / self.train_y.shape[0]

    def fit_params(self, n_iters):
        optim = torch.optim.Adam(self.parameters(), lr=0.1)
        for epoch in range(n_iters):
            optim.zero_grad()
            loss = self.log_marginal_likelihood()
            loss.backward()
            optim.step()

            # if epoch % 20 == 0:
            #     print(f'Epoch {epoch}, Log-Likelihood: {loss.item()}')
        print(f'Final Log-Likelihood: {loss.item():.5g}')

        kern_scale, kern_len, noise = self.param_holder.get_params()
        c_print(f'noise = {noise.item():.3g}', color="green")
        c_print(f'kern len = {kern_len.item():.3g}', color="green")
        c_print(f'kern var = {kern_scale.item():.3g}', color="green")

    def predict(self, pred_x):
        # print(pred_x.dim)
        assert pred_x.dim() == 2, "pred_x must be a 2D tensor."
        if not self.is_fitted:
            raise RuntimeError("The model must be fitted before prediction.")

        _, _, noise = self.param_holder.get_params()

        # Compute the cross-covariance matrix K_*
        K_star = self.kernel(pred_x, self.train_x)

        # Predict mean
        pred_mean = K_star @ self.inv_y

        # Predict covariance
        v = torch.linalg.solve_triangular(self.L, K_star.T, upper=False)
        pred_cov = self.kernel(pred_x, pred_x) - v.T @ v

        # Diagoanl variance
        diag_var = torch.diag(pred_cov)
        # Confidence region including observation noise
        conf_region = (pred_mean.squeeze() - 2 * torch.sqrt(diag_var + noise), pred_mean.squeeze() + 2 * torch.sqrt(diag_var +noise))
        preds = {"mean": pred_mean.squeeze(), "var": diag_var, "cov_matrix": pred_cov, 'conf_region': conf_region}
        return preds

    def predict_and_plot(self, test_x):
        with torch.no_grad():
            preds = self.predict(test_x.view(-1, 1))

        test_mean = preds['mean']
        lower, upper = preds['conf_region']

        plt.scatter(self.train_x, self.train_y)
        plt.plot(test_x.squeeze(), test_mean)
        plt.fill_between(test_x.squeeze(), lower, upper, alpha=0.5)
        plt.show()

        # x = torch.tensor([[0.25]], requires_grad=True)
        #
        # y = self.predict(x)['mean']
        # print(y)
        #
        # dydx = torch.autograd.grad(y, x, create_graph=True)[0]
        # print(dydx)


def pred_GPR(train_x, train_y, test_x):
    model = GPR(noise_var=None)
    model.fit(train_x.view(-1, 1), train_y, n_iters=100)

    model.predict_and_plot(test_x)


def pred_GPR_simple(train_x, train_y, test_x):
    model = GPRSimple(noise_var=None)
    model.fit(train_x.view(-1, 1), train_y, n_iters=100)
    # model.fit_params(n_iters=100)

    model.predict_and_plot(test_x)


def main():
    train_x = torch.linspace(0, 1, 100)
    train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.001)
    test_x = torch.linspace(0, 1, 51).view(-1, 1)

    pred_GPR(train_x, train_y, test_x)
    print()
    pred_GPR_simple(train_x, train_y, test_x)

    # out = model.predict(test_x)
    #
    # test_mean = out['mean']
    # lower, upper = out['conf_region']
    #
    # plt.scatter(train_x, train_y)
    # plt.plot(test_x, test_mean)
    # plt.fill_between(test_x, lower, upper, alpha=0.5)
    # plt.show()


if __name__ == "__main__":
    main()
