import torch
import math
import gpytorch
import matplotlib.pyplot as plt

# ---- Generate synthetic time series ----

# Time points
t = torch.linspace(0, 75, 500)


# Components
trend = 0.1 * t # linear trend
seasonality = torch.sin(2 * math.pi * t / 12) # yearly-like cycle


# Multiple Gaussian bumps
peak1 = torch.exp(-0.5 * ((t - 15) / 7) ** 2) * 4
peak2 = torch.exp(-0.5 * ((t - 30) / 5) ** 2) * 3
peak3 = torch.exp(-0.5 * ((t - 65) / 2) ** 2) * 2
peaks = peak1 + peak2 + peak3


# More random noise (heteroscedastic-like)
noise = 0.3 * torch.randn(t.size()) * (1 + 0.5 * torch.rand(t.size()))


# Full time series
y = trend + seasonality + noise + peaks


# Training data (subset)
train_x = t
train_y = y

# ---- Define GP model ----

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean() # captures trend
        # Kernel = RBF (smooth) + Periodic (seasonality) + RBF (local peaks)
        periodic_kernel = gpytorch.kernels.PeriodicKernel()
        periodic_kernel.initialize(period_length=12.0)


        self.covar_module = (
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
             + gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            + gpytorch.kernels.ScaleKernel(periodic_kernel)
        )


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# ---- Train GP ----

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x.unsqueeze(-1), train_y, likelihood)

model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss_fn = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 200
for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x.unsqueeze(-1))
    loss = -loss_fn(output, train_y)
    loss.backward()
    if i % 50 == 0:
        print(f'Iter {i+1}/{training_iter} - Loss: {loss.item():.3f}')
    optimizer.step()

# ---- Evaluate ----

model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = torch.linspace(0, 100, 600)
    preds = likelihood(model(test_x.unsqueeze(-1)))

# ---- Plot ----

mean = preds.mean
lower, upper = preds.confidence_region()

plt.figure(figsize=(12, 6))
plt.plot(t, y, 'k.', alpha=0.4, label='Observed Data')
plt.plot(test_x.numpy(), mean.numpy(), 'b', label='GP Mean')
plt.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.3, label='95% CI')
plt.legend()
plt.tight_layout()
plt.autoscale(tight=True)
plt.savefig("./Images/EndOfSummerPresentation/TSPredDemoLinearPeriodicRBF.pdf")
plt.show()
