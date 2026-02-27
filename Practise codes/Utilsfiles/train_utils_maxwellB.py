import os
import numpy as np
import torch
import jax.numpy as jnp
import flax
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from scipy.stats import gaussian_kde

def load_and_normalize_data_maxwellB(X_path, Y_path, seed=42):
    rng = np.random.default_rng(seed)

    # Load data (Physical Units)
    X = torch.load(X_path).numpy()
    Y = torch.load(Y_path).numpy()

    n_samples = X.shape[0]
    n_outputs = Y.shape[1]  # should be 6

    idx_train_set, idx_val_set, idx_test_set = set(), set(), set()

    # Per-component balanced split
    for comp in range(n_outputs):
        sorted_idx = np.argsort(np.abs(Y[:, comp]))[::-1]  # descending
        train_idx = sorted_idx[0::3]
        val_idx   = sorted_idx[1::3]
        test_idx  = sorted_idx[2::3]
        idx_train_set.update(train_idx)
        idx_val_set.update(val_idx)
        idx_test_set.update(test_idx)

    idx_train = list(idx_train_set)
    idx_val   = list(idx_val_set)
    idx_test  = list(idx_test_set)

    rng.shuffle(idx_train)
    rng.shuffle(idx_val)
    rng.shuffle(idx_test)

    X_train, Y_train = X[idx_train], Y[idx_train]
    X_val,   Y_val   = X[idx_val],   Y[idx_val]
    X_test,  Y_test  = X[idx_test],  Y[idx_test]

    # Normalization using TRAIN stats
    X_mean = X_train.mean(axis=0)
    X_std  = X_train.std(axis=0)
    X_std[X_std == 0] = 1
    Y_mean = Y_train.mean(axis=0)
    Y_std  = Y_train.std(axis=0)
    Y_std[Y_std == 0] = 1

    X_train_n = (X_train - X_mean) / X_std
    X_val_n   = (X_val   - X_mean) / X_std
    X_test_n  = (X_test  - X_mean) / X_std
    Y_train_n = (Y_train - Y_mean) / Y_std
    Y_val_n   = (Y_val   - Y_mean) / Y_std
    Y_test_n  = (Y_test  - Y_mean) / Y_std

    # Debug output
    def print_stats(name, arr):
        print(f"{name}  Y (Physical Units):")
        print("  Min :", arr.min(axis=0))
        print("  Max :", arr.max(axis=0))
        print("  Std :", arr.std(axis=0))

    print("\n=== Normalisation Sanity Check (MaxwellB Per-Component Balanced Split) ===")
    print_stats("Train", Y_train)
    print_stats("Val  ", Y_val)
    print_stats("Test ", Y_test)

    return (jnp.array(X_train_n), jnp.array(X_val_n), jnp.array(X_test_n),
            jnp.array(Y_train_n), jnp.array(Y_val_n), jnp.array(Y_test_n),
            X_mean, X_std, Y_mean, Y_std)

# ---------------------------
# Model checkpoint utilities
# ---------------------------

def save_checkpoint(params, X_mean, X_std, Y_mean, Y_std, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    to_save = {
        "params": params,
        "X_mean": X_mean,
        "X_std": X_std,
        "Y_mean": Y_mean,
        "Y_std": Y_std
    }
    with open(path, "wb") as f:
        f.write(flax.serialization.to_bytes(to_save))

def load_checkpoint(path, init_params):
    with open(path, "rb") as f:
        restored = flax.serialization.from_bytes(init_params, f.read())
    return restored

# ---------------------------
# Plotting utilities
# ---------------------------

def plot_learning_curves(train_losses, val_losses, fig_dir, model_type):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Learning Curves ({model_type})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "learning_curves.png"))
    plt.close()

def plot_residual_hist(residuals, fig_dir, model_type):
    residuals_1d = np.ravel(residuals)
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals_1d, bins=30, color='skyblue', stat='density')
    kde = gaussian_kde(residuals_1d)
    x_range = np.linspace(residuals_1d.min(), residuals_1d.max(), 1000)
    plt.plot(x_range, kde(x_range), color='orange', lw=2, label='KDE')
    plt.axvline(0, color='red', linestyle='--', label='Zero Residual')
    plt.title(f"Residuals on Test Data ({model_type})")
    plt.xlabel("Residual (True - Predicted)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "residual_histogram.png"))
    plt.close()

def plot_residuals_vs_pred(y_pred, residuals, fig_dir, model_type):
    y_pred_1d = np.ravel(y_pred)
    residuals_1d = np.ravel(residuals)
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred_1d, residuals_1d, alpha=0.5)
    smoothed = sm.nonparametric.lowess(residuals_1d, y_pred_1d, frac=0.3)
    plt.plot(smoothed[:, 0], smoothed[:, 1], color='red', lw=2, label='LOWESS')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residual (True - Predicted)")
    plt.title(f"Residuals vs Predicted ({model_type})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "residuals_vs_predictions.png"))
    plt.close()

# ---------------------------
# New: Stress Tensor Heatmap
# ---------------------------

def plot_stress_tensor_comparison(vec6_to_sym3, y_true_phys, y_pred_phys, sample_indices, save_dir, model_type):
    """
    Plot and save side-by-side true vs predicted symmetric 3x3 stress tensors for given sample indices.
    """
    import jax.numpy as jnp
    os.makedirs(save_dir, exist_ok=True)
    
    for idx in sample_indices:
        # Convert to 3x3 tensors
        T_true = np.array(vec6_to_sym3(jnp.array([y_true_phys[idx]]))).squeeze()
        T_pred = np.array(vec6_to_sym3(jnp.array([y_pred_phys[idx]]))).squeeze()

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # True tensor heatmap
        im0 = axes[0].imshow(T_true, cmap="viridis")
        for (i, j), val in np.ndenumerate(T_true):
            axes[0].text(j, i, f"{val:.2e}", ha='center', va='center', color='white')
        axes[0].set_title(f"True Stress Tensor (sample {idx})")
        fig.colorbar(im0, ax=axes[0], format="%.0e")

        # Predicted tensor heatmap
        im1 = axes[1].imshow(T_pred, cmap="viridis")
        for (i, j), val in np.ndenumerate(T_pred):
            axes[1].text(j, i, f"{val:.2e}", ha='center', va='center', color='white')
        axes[1].set_title(f"Predicted Stress Tensor (sample {idx})")
        fig.colorbar(im1, ax=axes[1], format="%.0e")

        plt.suptitle(f"{model_type} Stress Tensor Comparison")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"stress_tensor_comparison_sample_{idx}.png"))
        plt.close()