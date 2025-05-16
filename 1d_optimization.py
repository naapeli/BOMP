import matplotlib.pyplot as plt
import torch
from torch.optim import Adam, LBFGS
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel, PolynomialKernel
from gpytorch.constraints import Interval, Positive
from gpytorch.distributions import MultivariateNormal
from gpytorch.settings import fast_pred_var
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import LogProbabilityOfImprovement, ExpectedImprovement, LogExpectedImprovement, ProbabilityOfImprovement
from botorch.acquisition import qExpectedImprovement, qAnalyticProbabilityOfImprovement, qKnowledgeGradient, qLogNoisyExpectedImprovement, qLogExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

# ==================== DATA ====================
torch.manual_seed(1)
xmin, xmax, N = -5, 4, 500  # originally xmax = 5
true_coeffs = [0.0, 3.488378906, 0.0, -0.855187500, 0.0, 0.107675000, 0.0, -0.005857143, 0.0, 0.000111111]
# X = torch.linspace(xmin, xmax, N, dtype=torch.float64).unsqueeze(-1)
# m = 400
# X1 = torch.linspace(xmin, 2, m, dtype=torch.float64).unsqueeze(-1)
# X2 = torch.linspace(3, xmax, N - m, dtype=torch.float64).unsqueeze(-1)
# X = torch.cat([X1, X2], dim=0)
X = torch.rand(size=(N, 1), dtype=torch.float64) * (xmax - xmin) + xmin
y = sum(c * X ** i for i, c in enumerate(true_coeffs))
mask = -(X - xmin) * (X - xmax) / 50
y_true = y * mask
y = y_true + torch.normal(0, 1, size=(N, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9)
# feature_scaler = StandardScaler()
# X_train = torch.from_numpy(feature_scaler.fit_transform(X_train))
# X_test = torch.from_numpy(feature_scaler.transform(X_test))
target_scaler = RobustScaler()  # StandardScaler()
y_train = torch.from_numpy(target_scaler.fit_transform(y_train))
y_test = torch.from_numpy(target_scaler.transform(y_test))

# ==================== COVARIANCE AND MEAN ====================
# kernel = ScaleKernel(MaternKernel(nu=2.5, lengthscale_constraint=Positive()))
# kernel = ScaleKernel(RBFKernel(ard_num_dims=1, lengthscale_constraint=Positive()))
kernel = ScaleKernel(RBFKernel(ard_num_dims=1, lengthscale_constraint=Interval(1e-2, 1e1)))
# kernel = ScaleKernel(RBFKernel(ard_num_dims=1, lengthscale_constraint=Interval(1e-1, 1e0)))
# kernel = ScaleKernel(PolynomialKernel(power=5))
mean = ZeroMean()

# ==================== MODEL ====================
class ExactGaussianProcess(ExactGP):
    def __init__(self, X_train, y_train, likelihood):
        super().__init__(X_train, y_train, likelihood)
        self.mean_module = mean
        self.covar_module = kernel
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

bo_torch = 1
if bo_torch == 0:
    y_train = y_train.squeeze()
    likelihood = GaussianLikelihood()
    model = ExactGaussianProcess(X_train, y_train, likelihood).to(X.dtype)
elif bo_torch == 1:
    model = SingleTaskGP(X_train, y_train, covar_module=kernel, mean_module=mean)

# ==================== OBJECTIVE ====================
mll = ExactMarginalLogLikelihood(model.likelihood, model)

# ==================== TRAIN ====================
optimizer = Adam(model.parameters(), lr=0.001)
model.train()
model.likelihood.train()
if bo_torch == 0:
    for i in range(4000):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train).mean()
        loss.backward()
        optimizer.step()
        # if (i + 1) % 100 == 0: print(i + 1, loss.round(decimals=3).item())
elif bo_torch == 1:
    fit_gpytorch_mll(mll)

# ==================== PLOT ====================
X_test, _ = torch.sort(X_test, dim=0)
with torch.no_grad(), fast_pred_var():
    model.eval()
    model.likelihood.eval()

    if bo_torch == 0:
        post = model.likelihood(model(X_test))
    elif bo_torch == 1:
        post = model.posterior(X_test)
    mean = torch.from_numpy(target_scaler.inverse_transform(post.mean.reshape(-1, 1))).squeeze()
    stddev = post.stddev * target_scaler.scale_[0]
    # mean = post.mean.squeeze()
    # stddev = post.stddev
    lower, upper = mean - 1.96 * stddev, mean + 1.96 * stddev

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(X_train, torch.from_numpy(target_scaler.inverse_transform(y_train.reshape(-1, 1))).squeeze(), ".", label="Training data")
# plt.plot(test, mean, label="Predictions")
# plt.fill_between(test.squeeze(), lower.squeeze(), upper.squeeze(), color='gray', alpha=0.5, label="95% Confidence Interval")
# plt.plot(X_train, y_train.squeeze(), ".", label="Training data")
plt.plot(X_test, mean, label="Predictions")
temp, indicies = torch.sort(X, dim=0)
indicies = indicies.squeeze()
plt.plot(temp, y_true[indicies], label="True curve")
plt.fill_between(X_test.squeeze(), lower.squeeze(), upper.squeeze(), color='gray', alpha=0.5, label="95% Confidence Interval")


BATCH_SIZE = 1
NUM_RESTARTS = 10
RAW_SAMPLES = 200

# acq_func = qLogExpectedImprovement(model=model, best_f=y_train.max(), sampler=SobolQMCNormalSampler(torch.Size([1])),)
acq_func = qLogNoisyExpectedImprovement(model, X_train, sampler=SobolQMCNormalSampler(torch.Size([1])))
# acq_func = LogProbabilityOfImprovement(model, y_train.max())

candidates, _ = optimize_acqf(
    acq_function=acq_func,
    bounds=torch.tensor([[xmin], [xmax]], dtype=torch.float),
    q=BATCH_SIZE,
    num_restarts=NUM_RESTARTS,
    raw_samples=RAW_SAMPLES,
    # options={"batch_limit": 5, "maxiter": 200},
    sequential=True,
)
for item in candidates.squeeze(-1):
    plt.axvline(item.numpy(), label="Optimal next point")
plt.legend()


plt.subplot(2, 1, 2)
X_acq_plot = torch.linspace(xmin, xmax, 1000).unsqueeze(-1)
# X_acq_scaled = torch.from_numpy(feature_scaler.transform(X_acq_plot.numpy()))
acq_func.eval()
X_acq_plot = torch.linspace(xmin, xmax, 500, dtype=torch.float64).unsqueeze(-1)
X_batch = X_acq_plot.unsqueeze(1)
with torch.no_grad():
    acq_vals = acq_func(X_batch).squeeze()
plt.plot(X_acq_plot.numpy(), acq_vals.numpy(), label="Acquisition Function (e.g., qLogEI)")
plt.xlabel("Input")
plt.ylabel("Acquisition Value")
plt.title("Acquisition Function Landscape")
plt.grid(True)
plt.legend()

plt.show()
