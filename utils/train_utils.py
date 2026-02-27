import os
import numpy as np
import torch
import jax
import jax.numpy as jnp
import flax
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score, mean_absolute_error

def load_and_normalize_data(X_path, Y_path, seed=42, test_size=0.1, val_size=0.1):
    X = torch.load(X_path).numpy()
    Y = torch.load(Y_path).numpy()
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=val_size, random_state=seed)

    # Normalize X
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std == 0] = 1
    X_train = (X_train - X_mean) / X_std
    X_val   = (X_val - X_mean) / X_std
    X_test  = (X_test - X_mean) / X_std

    # Normalize Y
    Y_mean = Y_train.mean(axis=0)
    Y_std = Y_train.std(axis=0)
    Y_std[Y_std == 0] = 1
    Y_train = (Y_train - Y_mean) / Y_std
    Y_val   = (Y_val - Y_mean) / Y_std
    Y_test  = (Y_test - Y_mean) / Y_std

    
    return (jnp.array(X_train), jnp.array(X_val), jnp.array(X_test),
            jnp.array(Y_train), jnp.array(Y_val), jnp.array(Y_test),
            X_mean, X_std, Y_mean, Y_std)

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
#***********************************************************************
# Plotting functions
#*******************************************************************
def plot_learning_curves(train_losses, val_losses,
                         train_data_losses, val_data_losses,
                         train_phys_losses, val_phys_losses,
                         fig_dir, model_type):
    import matplotlib.pyplot as plt

    # Smoothing helper
    def smooth_curve(values, weight=0.9):
        smoothed, last = [], values[0]
        for val in values:
            smoothed_val = last * weight + (1 - weight) * val
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    # Smooth all 6 curves
    ttl_train = smooth_curve(train_losses)
    ttl_val   = smooth_curve(val_losses)
    dat_train = smooth_curve(train_data_losses)
    dat_val   = smooth_curve(val_data_losses)
    phy_train = smooth_curve(train_phys_losses)
    phy_val   = smooth_curve(val_phys_losses)

    plt.figure(figsize=(10,6))

    # Plot each with distinct style & marker
    plt.plot(ttl_train, label="Train Total Loss", color='tab:blue', linewidth=2)
    plt.plot(ttl_val,   label="Val Total Loss",   color='tab:orange', linewidth=2)
    plt.plot(dat_train, label="Train Data Loss",    color='tab:green', linestyle='-.', marker='^', markersize=4, linewidth=2)
    plt.plot(dat_val,   label="Val Data Loss",      color='tab:red',   linestyle=':',  marker='d', markersize=4, linewidth=2)
    plt.plot(phy_train, label="Train Physics Loss", color='tab:purple',linestyle='-',  marker='x', markersize=4, linewidth=2)
    plt.plot(phy_val,   label="Val Physics Loss",   color='tab:brown', linestyle='--', marker='*', markersize=4, linewidth=2)

    plt.yscale("log")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("MSE Loss", fontsize=14)
    plt.title(f"All Losses ({model_type})", fontsize=16)
    plt.legend(fontsize=10, loc="upper right")
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.tight_layout()

    save_path = os.path.join(fig_dir, "learning_curves.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved all-losses plot to {save_path}")

def plot_residual_hist(residuals, fig_dir, model_type):
    residuals_1d = np.ravel(residuals)
    plt.figure(figsize=(8,5))
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
    plt.figure(figsize=(8,5))
    plt.scatter(y_pred_1d, residuals_1d, alpha=0.5)
    smoothed = sm.nonparametric.lowess(residuals_1d, y_pred_1d, frac=0.3)
    plt.plot(smoothed[:,0], smoothed[:,1], color='red', lw=2, label='LOWESS')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residual (True - Predicted)")
    plt.title(f"Residuals vs Predicted ({model_type})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "residuals_vs_predictions.png"))
    plt.close()

def plot_true_pred_phys_samples_with_values(
    y_true_phys_samples,
    y_pred_phys_samples,
    y_phys_samples,
    fig_dir,
    model_type,
    sample_indices=None
):
    """
    Plot True vs Predicted vs Physics Law AND Difference (True - Predicted).
    """
    
    num_samples = len(y_true_phys_samples)
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))  # now 4 columns

    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_samples):
        diff = y_true_phys_samples[i] - y_pred_phys_samples[i]

        data_list = [
            y_true_phys_samples[i].reshape(1, -1),
            y_pred_phys_samples[i].reshape(1, -1),
            y_phys_samples[i].reshape(1, -1),
            diff.reshape(1, -1)
        ]

        titles = [
            f"Sample {sample_indices[i] if sample_indices is not None else i} — True",
            "Predicted",
            "Physics Law",
            "True - Predicted"
        ]

        for j, data in enumerate(data_list):
            ax = axes[i, j]
            # For difference plot, use a separate colormap scale centered at 0
            if j == 3:  # Difference plot
                vmin = np.min(data_list[j])
                vmax = np.max(data_list[j])
                sns.heatmap(data, cmap="coolwarm", center=0, cbar=True, ax=ax)
            else:
                sns.heatmap(data, cmap="viridis", cbar=True, ax=ax)

            # Add value as text
            val = data.flatten()[0]
            ax.text(0.5, 0.5, f"{val:.2e}", color="white", ha="center", va="center",
                    fontsize=12, fontweight="bold")

            ax.set_title(titles[j])
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle(f"True vs Predicted vs Physics vs Difference ({model_type})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(fig_dir, f"true_pred_phys_diff_{model_type}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved comparison heatmaps + difference to {save_path}")