import matplotlib.pyplot as plt
import torch
from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel, PolynomialKernel
from gpytorch.constraints import Interval, Positive
from gpytorch.settings import fast_pred_var
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP, ModelListGP
from botorch.optim import optimize_acqf
from botorch.acquisition.multi_objective import qLogExpectedHypervolumeImprovement, qMultiFidelityHypervolumeKnowledgeGradient
from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning
from botorch.utils.multi_objective import is_non_dominated
from botorch.sampling import SobolQMCNormalSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

# ==================== DATA ====================
torch.manual_seed(1)
xmin, xmax, N = -5, 4, 500  # originally xmax = 5
true_coeffs1 = [0.0, 3.488378906, 0.0, -0.855187500, 0.0, 0.107675000, 0.0, -0.005857143, 0.0, 0.000111111]
true_coeffs2 = [0.0, 1.2, -0.0009, -0.28, 0.0003, 0.025, -0.00012, -0.0018, 0.00001, 0.000045]

dtype = torch.float64
X = torch.rand(size=(N, 1), dtype=dtype) * (xmax - xmin) + xmin
mask = -(X - xmin) * (X - xmax) / 50
y1 = sum(c * X ** i for i, c in enumerate(true_coeffs1))
y2 = sum(c * X ** i for i, c in enumerate(true_coeffs2))
y1_true = y1 * mask
y2_true = y2 * mask
y1 = y1_true + 0.1 * torch.normal(0, 1, size=(N, 1), dtype=dtype)
y2 = y2_true + 0.1 * torch.normal(0, 1, size=(N, 1), dtype=dtype)
y = torch.cat([y1, y2], dim=1)
# plt.plot(X, y1, ".")
# plt.plot(X, y2, ".")
# plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)
# feature_scaler = StandardScaler()
# X_train = torch.from_numpy(feature_scaler.fit_transform(X_train))
# X_test = torch.from_numpy(feature_scaler.transform(X_test))
target_scaler = RobustScaler()  # StandardScaler()
y_train = torch.from_numpy(target_scaler.fit_transform(y_train)).to(dtype=dtype)
y_test = torch.from_numpy(target_scaler.transform(y_test)).to(dtype=dtype)

# ==================== MODEL ====================
gps = []
for i in range(y_train.shape[1]):
    # kernel = ScaleKernel(MaternKernel(nu=2.5, lengthscale_constraint=Positive()))
    # kernel = ScaleKernel(RBFKernel(ard_num_dims=1, lengthscale_constraint=Positive()))
    kernel = ScaleKernel(PolynomialKernel(power=5))
    mean = ZeroMean()

    gps.append(SingleTaskGP(X_train, y_train[:, i:i+1], covar_module=kernel, mean_module=mean))

model = ModelListGP(*gps)

mll = SumMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

# ==================== PLOT ====================
X_test, indicies = torch.sort(X_test, dim=0)
indicies = indicies.squeeze()
with torch.no_grad(), fast_pred_var():
    model.eval()
    model.likelihood.eval()
    post = model.posterior(X_test)

BATCH_SIZE = 2
NUM_RESTARTS = 10
RAW_SAMPLES = 200

ref_point = torch.max(y_train, dim=0)[0]
partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=y_train)
acq_func = qLogExpectedHypervolumeImprovement(model, ref_point, partitioning, sampler=SobolQMCNormalSampler(ref_point.shape))

candidates, _ = optimize_acqf(
    acq_function=acq_func,
    bounds=torch.tensor([[xmin], [xmax]], dtype=torch.float),
    q=BATCH_SIZE,
    num_restarts=NUM_RESTARTS,
    raw_samples=RAW_SAMPLES,
    # options={"batch_limit": 5, "maxiter": 200},
    sequential=False,  # NOTE: if False, the candidates are mostly on 2 peaks as supposed to
)

plt.figure(figsize=(14, 8))

temp_test = torch.from_numpy(target_scaler.inverse_transform(y_test[indicies])).to(dtype=dtype)
prediction = torch.from_numpy(target_scaler.inverse_transform(post.mean)).to(dtype=dtype)
prediction_std = post.variance * torch.from_numpy(target_scaler.scale_)
temp_train = torch.from_numpy(target_scaler.inverse_transform(y_train)).to(dtype=dtype)

X_test_sorted, test_sort_idx = torch.sort(X_test.squeeze(), dim=0)

for i in range(2):
    plt.subplot(2, 3, i + 1)
    plt.scatter(temp_test[:, i].numpy(), prediction[:, i].numpy(), label=f'Objective {i + 1}')
    plt.plot([-2, 2], [-2, 2], 'k--', label="Perfect fit")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"True vs Predicted (Obj {i + 1})")
    plt.legend()

    plt.subplot(2, 3, i + 4)
    mean = prediction[:, i][test_sort_idx]
    stddev = prediction_std[:, i][test_sort_idx]
    lower, upper = mean - 1.96 * stddev, mean + 1.96 * stddev
    plt.plot(X_test_sorted.numpy(), mean.numpy(), label="Prediction")
    plt.fill_between(X_test_sorted.numpy(), lower.numpy(), upper.numpy(), alpha=0.3, label="95% CI")
    plt.plot(X_train.numpy(), temp_train[:, i].numpy(), ".", label="Training Data")

    for item in candidates.squeeze(-1):
        plt.axvline(item.item(), color='red', linestyle='--')
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title(f"Prediction + CI (Obj {i + 1})")

    temp, idx = torch.sort(X.squeeze(), dim=0)
    true_curve = y1_true if i == 0 else y2_true
    plt.plot(temp.numpy(), true_curve[idx].numpy(), label="True curve")
    plt.legend()

plt.subplot(2, 3, 6)
X_acq_plot = torch.linspace(xmin, xmax, 500, dtype=dtype).unsqueeze(-1)
X_batch = X_acq_plot.unsqueeze(1)
with torch.no_grad():
    acq_vals = acq_func(X_batch).squeeze()
plt.plot(X_acq_plot.numpy(), acq_vals.numpy())
plt.title("qLogEHVI Acquisition Function")
plt.xlabel("Input")
plt.ylabel("Acquisition Value")

with torch.no_grad():
    candidate_post = model.posterior(candidates)
    candidate_mean = candidate_post.mean
    candidate_mean_unscaled = torch.from_numpy(
        target_scaler.inverse_transform(candidate_mean)
    ).to(dtype=dtype)

y_train_unscaled = temp_train
pareto_mask = is_non_dominated(y_train_unscaled)
pareto_front = y_train_unscaled[pareto_mask]

plt.subplot(2, 3, 3)
plt.scatter(y_train_unscaled[:, 0], y_train_unscaled[:, 1], alpha=0.3, label="All points")
plt.scatter(pareto_front[:, 0], pareto_front[:, 1], color='blue', label="Pareto front")
plt.scatter(candidate_mean_unscaled[:, 0], candidate_mean_unscaled[:, 1],
                  color='red', marker='x', s=100, label="New candidates")
plt.xlabel("Objective 1")
plt.ylabel("Objective 2")
plt.title("Pareto Front with New Candidates")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("Images/multi_objective_optimization.png")
plt.show()
