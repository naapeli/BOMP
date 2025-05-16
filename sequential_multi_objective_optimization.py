import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.animation import FuncAnimation
import numpy as np
import torch
import warnings
from time import perf_counter
from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel, PolynomialKernel
from gpytorch.constraints import Interval, Positive
from gpytorch.settings import fast_pred_var
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP, ModelListGP
from botorch.optim import optimize_acqf
from botorch.acquisition.multi_objective import qLogExpectedHypervolumeImprovement, qHypervolumeKnowledgeGradient
from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning, DominatedPartitioning
from botorch.sampling import SobolQMCNormalSampler


warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==================== DATA ====================
torch.manual_seed(1)
xmin, xmax = -5, 4
dtype = torch.float64
true_coeffs1 = [0.0, 3.488378906, 0.0, -0.855187500, 0.0, 0.107675000, 0.0, -0.005857143, 0.0, 0.000111111]
true_coeffs2 = [0.0, 1.2, -0.0009, -0.28, 0.0003, 0.025, -0.00012, -0.0018, 0.00001, 0.000045]

def f(x):
    mask = -(x - xmin) * (x - xmax) / 50
    return sum(c * x ** i for i, c in enumerate(true_coeffs1)) * mask

def g(x):
    mask = -(x - xmin) * (x - xmax) / 50
    return sum(c * x ** i for i, c in enumerate(true_coeffs2)) * mask

def add_noise(y):
    return y + torch.normal(0, 0.1, size=y.shape, dtype=dtype)

def true_data(x):
    return torch.cat([f(x), g(x)], dim=1)

def model_forward(x):
    return add_noise(torch.cat([f(x), g(x)], dim=1))

def generate_data(n):
    x = torch.rand(size=(n, 1), dtype=dtype) * (xmax - xmin) + xmin
    y = true_data(x)
    y_noisy = add_noise(y)
    return x, y, y_noisy

# ==================== MODEL ====================
def initialize_model(X, y):
    gps = []
    for i in range(y.shape[1]):
        # kernel = ScaleKernel(MaternKernel(nu=2.5, lengthscale_constraint=Positive()))
        kernel = ScaleKernel(RBFKernel(ard_num_dims=1, lengthscale_constraint=Positive()))
        # kernel = ScaleKernel(PolynomialKernel(power=5))
        mean = ZeroMean()
        gps.append(SingleTaskGP(X, y[:, i:i+1], covar_module=kernel, mean_module=mean))

    model = ModelListGP(*gps)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model

# ==================== OPTIMIZE ====================
def optimize_acq_func_and_get_observation(acq_func):
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=torch.tensor([[xmin], [xmax]], dtype=torch.float),
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        # options={"batch_limit": 5, "maxiter": 200},
        sequential=False  # as qHypervolumeKnowledgeGradient is a one-shot acquisition function, must use sequential=False
    )

    new_obj_true = model_forward(candidates)
    new_obj = add_noise(new_obj_true)
    return candidates, new_obj, new_obj_true


BATCH_SIZE = 1
NUM_RESTARTS = 10
RAW_SAMPLES = 200
N_ROUNDS = 40

x_dense = torch.linspace(xmin, xmax, 100000).unsqueeze(-1)
Y_dense = model_forward(x_dense)
ref_point = Y_dense.min(dim=0)[0]
bd_dense = DominatedPartitioning(ref_point=ref_point, Y=Y_dense)
max_hv = bd_dense.compute_hypervolume().item()
print(max_hv)

n_init = 1
X_EI, y_true_EI, y_EI = generate_data(n_init)
X_random, y_true_random, y_random = X_EI.clone(), y_true_EI.clone(), y_EI.clone()
X_KG, y_true_KG, y_KG = X_EI.clone(), y_true_EI.clone(), y_EI.clone()

x_plot = torch.linspace(xmin, xmax, 500, dtype=dtype).unsqueeze(-1)

hvs_EI, hvs_KG, hvs_random = [], [], []
batch_number_random = torch.zeros(n_init)
batch_number_EI = torch.zeros(n_init)
batch_number_KG = torch.zeros(n_init)
predictions_EI, predictions_KG = [], []

mll_EI, model_EI = initialize_model(X_EI, y_EI)
mll_KG, model_KG = initialize_model(X_EI, y_EI)
fit_gpytorch_mll(mll_EI)
fit_gpytorch_mll(mll_KG)
model_EI.eval()
model_KG.eval()
with torch.no_grad(), fast_pred_var():
    predictions_EI.append(model_EI.posterior(x_plot))
    predictions_KG.append(model_KG.posterior(x_plot))
model_EI.train()
model_KG.train()

bd = DominatedPartitioning(ref_point=ref_point, Y=y_EI)
volume = bd.compute_hypervolume().item()

hvs_EI.append(volume)
hvs_random.append(volume)
hvs_KG.append(volume)

for iteration in range(1, N_ROUNDS + 1):
    start = perf_counter()
    fit_gpytorch_mll(mll_EI)

    acq_func_EI = qLogExpectedHypervolumeImprovement(model_EI, ref_point, FastNondominatedPartitioning(ref_point=ref_point, Y=y_EI), sampler=SobolQMCNormalSampler(ref_point.shape))
    acq_func_KG = qHypervolumeKnowledgeGradient(model_KG, ref_point, num_fantasies=8)
    new_x_EI, new_obj_EI, new_obj_true_EI = optimize_acq_func_and_get_observation(acq_func_EI)
    new_x_KG, new_obj_KG, new_obj_true_KG = optimize_acq_func_and_get_observation(acq_func_KG)
    new_x_random, new_obj_true_random, new_obj_random = generate_data(BATCH_SIZE)

    X_EI = torch.cat([X_EI, new_x_EI])
    y_EI = torch.cat([y_EI, new_obj_EI])
    y_true_EI = torch.cat([y_true_EI, new_obj_true_EI])

    X_random = torch.cat([X_random, new_x_random])
    y_random = torch.cat([y_random, new_obj_random])
    y_true_random = torch.cat([y_true_random, new_obj_true_random])

    X_KG = torch.cat([X_KG, new_x_KG])
    y_KG = torch.cat([y_KG, new_obj_KG])
    y_true_KG = torch.cat([y_true_KG, new_obj_true_KG])

    for hvs_list, train_obj in zip((hvs_random, hvs_EI, hvs_KG), (y_random, y_EI, y_KG)):
        bd = DominatedPartitioning(ref_point=ref_point, Y=train_obj)
        volume = bd.compute_hypervolume().item()
        hvs_list.append(volume)

    mll_EI, model_EI = initialize_model(X_EI, y_EI)
    mll_KG, model_KG = initialize_model(X_EI, y_EI)

    model_EI.eval()
    model_KG.eval()
    with torch.no_grad(), fast_pred_var():
        predictions_EI.append(model_EI.posterior(x_plot))
        predictions_KG.append(model_KG.posterior(x_plot))
    model_EI.train()
    model_KG.train()
    
    batch_number_random = torch.cat([batch_number_random, torch.full((BATCH_SIZE,), iteration)])
    batch_number_EI = torch.cat([batch_number_EI, torch.full((BATCH_SIZE,), iteration)])
    batch_number_KG = torch.cat([batch_number_KG, torch.full((BATCH_SIZE,), iteration)])

    print(f"Batch {iteration:>2}: Hypervolume (random, qEHVI, qHVKG) = ({hvs_random[-1]:>4.2f}, {hvs_EI[-1]:>4.2f}, {hvs_KG[-1]:>4.2f}) - Total points = {len(batch_number_EI)} - Time for iteration = {perf_counter() - start:.2f}")



iters = np.arange(N_ROUNDS + 1) * BATCH_SIZE
log_hv_diff_EI = np.log10(max_hv - np.asarray(hvs_EI))
log_hv_diff_random = np.log10(max_hv - np.asarray(hvs_random))
log_hv_diff_KG = np.log10(max_hv - np.asarray(hvs_KG))

plt.figure(figsize=(8, 6))
plt.plot(iters, log_hv_diff_random, label="Random", linestyle='--', marker='o')
plt.plot(iters, log_hv_diff_EI, label="qLEHVI", linestyle='-', marker='s')
plt.plot(iters, log_hv_diff_KG, label="qHVKG", linestyle='-', marker='^')
plt.xlabel("Number of observations (beyond initial points)")
plt.ylabel("Log Hypervolume Difference")
plt.legend(loc="lower left")
plt.grid(True)
plt.tight_layout()
plt.savefig("sequential_hypervolumes.png")


fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharex=True, sharey=True)
algos = ["Random", "qEHVI", "qHVKG"]
cm = plt.get_cmap("viridis")

for i, (train_obj, batch_num) in enumerate(((y_true_random, batch_number_random), (y_true_EI, batch_number_EI), (y_true_KG, batch_number_KG))):
    sc = axes[i].scatter(
        train_obj[:, 0].numpy(),
        train_obj[:, 1].numpy(),
        c=batch_num.numpy(),
        cmap=cm,
        alpha=0.8,
    )
    axes[i].set_title(algos[i])
    axes[i].set_xlabel("Objective 1")

axes[0].set_ylabel("Objective 2")

norm = plt.Normalize(batch_number_EI.min().item(), batch_number_EI.max().item())
sm = ScalarMappable(norm=norm, cmap=cm)
sm.set_array([])

fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.9, 0.15, 0.015, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.set_title("Iteration")
plt.savefig("sequential_samples.png")


n_iters = N_ROUNDS + 1
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
axes = axes.ravel()
titles = ["Objective 1", "Objective 2"]
suptitle = fig.suptitle("")

def update(frame):
    suptitle.set_text(f"Iteration {frame}")

    for ax in axes:
        ax.clear()

    pred = predictions_EI[frame]
    mean = pred.mean
    std = pred.variance.sqrt()
    lower = mean - 1.96 * std
    upper = mean + 1.96 * std
    means = [mean[:, i] for i in range(mean.shape[1])]
    lower = [lower[:, i] for i in range(lower.shape[1])]
    upper = [upper[:, i] for i in range(upper.shape[1])]
    X_frame = X_EI[: n_init + frame * BATCH_SIZE + 1]
    y_frame = y_EI[: n_init + frame * BATCH_SIZE + 1]

    for i, ax in enumerate(axes[:2]):
        ax.scatter(X_frame, y_frame[:, i], c="k", label="Observations")
        ax.plot(x_plot, means[i], label="GP mean with qLEHVI")
        ax.plot(x_plot, true_data(x_plot)[:, i], label="True function")
        ax.fill_between(x_plot.squeeze(), lower[i], upper[i], alpha=0.3, label="Confidence")
        ax.set_ylabel(titles[i])
        ax.legend(loc="upper right")
        ax.grid(True)
        ax.set_ylim(-3, 3)
    
    pred = predictions_KG[frame]
    mean = pred.mean
    std = pred.variance.sqrt()
    lower = mean - 1.96 * std
    upper = mean + 1.96 * std
    means = [mean[:, i] for i in range(mean.shape[1])]
    lower = [lower[:, i] for i in range(lower.shape[1])]
    upper = [upper[:, i] for i in range(upper.shape[1])]
    X_frame = X_KG[: n_init + frame * BATCH_SIZE + 1]
    y_frame = y_KG[: n_init + frame * BATCH_SIZE + 1]

    for i, ax in enumerate(axes[2:]):
        ax.scatter(X_frame, y_frame[:, i], c="k", label="Observations")
        ax.plot(x_plot, means[i], label="GP mean with qHVKG")
        ax.plot(x_plot, true_data(x_plot)[:, i], label="True function")
        ax.fill_between(x_plot.squeeze(), lower[i], upper[i], alpha=0.3, label="Confidence")
        ax.set_ylabel(titles[i])
        ax.legend(loc="upper right")
        ax.grid(True)
        ax.set_ylim(-3, 3)
        ax.set_xlabel("Input x")

ani = FuncAnimation(fig, update, frames=n_iters, interval=100)
ani.save("sequential_animation.gif", writer="pillow")
plt.show()
