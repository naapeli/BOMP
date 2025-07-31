import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# --- True function ---
def true_function(x):
    return np.sin(x) + 0.2*np.cos(5*x)

# Training data (sample points)
# X_train = np.array([[0.5], [1.5], [2.5], [3.5], [4.5], [5.5]])
np.random.seed(6)
X_train = np.random.uniform(high=6, size=(5, 1))
y_train = true_function(X_train).ravel()

# Test points for plotting
X = np.linspace(0, 6, 400).reshape(-1, 1)
y_true = true_function(X)

# Gaussian Process with RBF kernel
kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-4)
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, optimizer=None)
gp.fit(X_train, y_train)

# GP predictions
y_mean, y_std = gp.predict(X, return_std=True)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(X, y_true, 'k-', label='True function', linewidth=2)
plt.plot(X, y_mean, 'b-', label='GP mean', linewidth=2)
plt.fill_between(X.ravel(), y_mean - 2*y_std, y_mean + 2*y_std,
                 color='blue', alpha=0.2, label='Uncertainty region (±2σ)')
plt.scatter(X_train, y_train, color='red', s=60, label='Sampled points', zorder=5)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./Images/July presentation/sausage_plot.pdf")
plt.show()
