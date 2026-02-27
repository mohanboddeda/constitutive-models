import os
import numpy as np
import torch
import jax.numpy as jnp
import flax
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import gaussian_kde



def balanced_split_quantiles(X, Y, train_frac=0.6, val_frac=0.2, test_frac=0.2,
                             n_bins=10, seed=42):
    """
    Stratified balanced split based on Frobenius norm magnitude using quantile bins.

    Parameters
    ----------
    X : np.ndarray
        Shape (N, features)
    Y : np.ndarray
        Shape (N, 6) -> symmetric tensor components: [Txx, Tyy, Tzz, Txy, Txz, Tyz]
    train_frac, val_frac, test_frac : float
        Fractions for train/val/test (should sum to 1.0)
    n_bins : int
        Number of quantile bins across Frobenius magnitude.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    X_train, Y_train, X_val, Y_val, X_test, Y_test
    """

    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-8, "Fractions must sum to 1."

    # Compute Frobenius norm magnitude from 6-component vector
    magnitudes = np.sqrt(
        Y[:, 0]**2 + Y[:, 1]**2 + Y[:, 2]**2 +
        2*(Y[:, 3]**2 + Y[:, 4]**2 + Y[:, 5]**2)
    )

    rng = np.random.default_rng(seed)

    # Quantile bin edges
    quantile_edges = np.quantile(magnitudes, np.linspace(0, 1, n_bins+1))

    train_idx, val_idx, test_idx = [], [], []

    for i in range(n_bins):
        bin_mask = (magnitudes >= quantile_edges[i]) & (magnitudes <= quantile_edges[i+1])
        bin_indices = np.where(bin_mask)[0]
        rng.shuffle(bin_indices)

        n_bin = len(bin_indices)
        # Apply fractions
        n_train = int(round(train_frac * n_bin))
        n_val   = int(round(val_frac * n_bin))
        n_test  = n_bin - n_train - n_val  # ensure all samples assigned

        train_idx.extend(bin_indices[:n_train])
        val_idx.extend(bin_indices[n_train:n_train+n_val])
        test_idx.extend(bin_indices[n_train+n_val:])

    # Shuffle final indices to avoid information leakage
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return (X[train_idx], Y[train_idx],
            X[val_idx], Y[val_idx],
            X[test_idx], Y[test_idx])
# =========================
# Main loader
# =========================
def load_and_normalize_data_maxwellB(
    X_path, Y_path, seed=42, test_size=0.2, val_size=0.2,
    balanced_split=False, scaling_mode="standard"
):
    """
    Loads X and Y from .pt files, splits into train/val/test,
    and applies normalization/scaling.

    Parameters
    ----------
    balanced_split : bool
        If True, performs quantile-magnitude stratified split.
    scaling_mode : str
        "standard" = mean/std normalization
        "minmax"   = min/max scaling to [0, 1]
    """

    rng = np.random.default_rng(seed)

    # ===== Load data (physical units) =====
    X = torch.load(X_path).numpy()
    Y = torch.load(Y_path).numpy()

    # ===== Split data =====
    if balanced_split:
        X_train, Y_train, X_val, Y_val, X_test, Y_test = \
            balanced_split_quantiles(X, Y, test_frac=test_size, val_frac=val_size, n_bins=10, seed=seed)
    else:
        from sklearn.model_selection import train_test_split
        X_train_val, X_test, Y_train_val, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=seed, shuffle=True
        )
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_train_val, Y_train_val, test_size=0.25, random_state=seed
        )

    # ===== X normalization (always mean/std) =====
    X_mean = X_train.mean(axis=0)
    X_std  = X_train.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_train_n = (X_train - X_mean) / X_std
    X_val_n   = (X_val   - X_mean) / X_std
    X_test_n  = (X_test  - X_mean) / X_std

    # ===== Y normalization/scaling =====
    if scaling_mode == "standard":
        Y_mean = Y_train.mean(axis=0)
        Y_std  = Y_train.std(axis=0)
        Y_std[Y_std == 0] = 1.0
        Y_train_n = (Y_train - Y_mean) / Y_std
        Y_val_n   = (Y_val   - Y_mean) / Y_std
        Y_test_n  = (Y_test  - Y_mean) / Y_std
    elif scaling_mode == "minmax":
        Y_min = Y_train.min(axis=0)
        Y_max = Y_train.max(axis=0)
        Y_range = np.where((Y_max - Y_min) == 0, 1.0, Y_max - Y_min)
        Y_train_n = (Y_train - Y_min) / Y_range
        Y_val_n   = (Y_val   - Y_min) / Y_range
        Y_test_n  = (Y_test  - Y_min) / Y_range
        # Store Y_min/Y_range to keep API consistent
        Y_mean, Y_std = Y_min, Y_range
    else:
        raise ValueError("scaling_mode must be either 'standard' or 'minmax'")

    # ===== Debug print: Stats in physical units =====
    def print_stats(name, arr):
        print(f"{name} Y (Physical Units):")
        print("  Min :", arr.min(axis=0))
        print("  Max :", arr.max(axis=0))
        print("  Std :", arr.std(axis=0))

    print("\n=== Normalisation Sanity Check (MaxwellB) ===")
    print_stats("Train", Y_train)
    print_stats("Val  ", Y_val)
    print_stats("Test ", Y_test)

    # ===== Debug print: Stats in normalized/scaled units =====
    def print_stats_norm(name, arr):
        print(f"{name} Y ({scaling_mode} scaled):")
        print("  Min :", arr.min(axis=0))
        print("  Max :", arr.max(axis=0))
        print("  Std :", arr.std(axis=0))

    print("\n--- Data after {} scaling ---".format(scaling_mode))
    print_stats_norm("Train", Y_train_n)
    print_stats_norm("Val  ", Y_val_n)
    print_stats_norm("Test ", Y_test_n)

    return (jnp.array(X_train_n), jnp.array(X_val_n), jnp.array(X_test_n),
            jnp.array(Y_train_n), jnp.array(Y_val_n), jnp.array(Y_test_n),
            X_mean, X_std, Y_mean, Y_std)

# =========================
# Checkpoint Utilities
# =========================
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
# =========================
# Plotting Utilities
# =========================
def plot_learning_curves_physical(train_losses, val_losses, Y_std, fig_dir, model_type):
    """
    Smoothly plot training/validation loss curves in physical units for Maxwell-B.
    Uses the average variance from Y_std to convert normalized MSE to physical MSE.
    """
    # Compute average variance across all output components
    Y_var = float(np.mean(np.array(Y_std)**2))

    # Convert normalized MSE → physical MSE
    train_losses_phys = np.array(train_losses) * Y_var
    val_losses_phys   = np.array(val_losses)   * Y_var

    # Smoothing helper
    def smooth_curve(values, weight=0.9):
        smoothed = []
        last = values[0]
        for val in values:
            smoothed_val = last * weight + (1 - weight) * val
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    train_plot = smooth_curve(train_losses_phys)
    val_plot   = smooth_curve(val_losses_phys)

    plt.figure(figsize=(8,5))
    plt.plot(train_plot, label="Training Loss (phys. units)", color='tab:blue', linewidth=2)
    plt.plot(val_plot, label="Validation Loss (phys. units)", color='tab:orange', linewidth=2)

    plt.yscale("log")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel(r"MSE Loss [Pa$^2$]", fontsize=14)  # Stress in Pascal → squared for MSE
    plt.title(f"Learning Curves ({model_type}) — Physical Units", fontsize=16)
    plt.legend(fontsize=12, loc="upper right")
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.tight_layout()

    save_path = os.path.join(fig_dir, "learning_curves_physical.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved smoothed physical-unit learning curve to {save_path}")

    

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


def plot_stress_tensor_comparison(vec6_to_sym3, y_true_phys, y_pred_phys, sample_indices, save_dir, model_type):
    os.makedirs(save_dir, exist_ok=True)
    eps = 1e-12  # small number to avoid divide-by-zero in relative error

    for idx in sample_indices:
        # Convert vector → symmetric 3x3 tensors
        T_true = np.array(vec6_to_sym3(jnp.array([y_true_phys[idx]]))).squeeze()
        T_pred = np.array(vec6_to_sym3(jnp.array([y_pred_phys[idx]]))).squeeze()
        T_err = T_true - T_pred
        T_relerr = np.abs(T_err) / (np.abs(T_true) + eps) * 100.0  # % relative error

        # Use same colormap scaling for True & Pred
        common_min = min(np.min(T_true), np.min(T_pred))
        common_max = max(np.max(T_true), np.max(T_pred))

        # Symmetric scale for absolute error colormap
        err_max = np.max(np.abs(T_err))

        # Create figure with 4 panels: True, Pred, Abs Error, Rel Error (%)
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))

        # --- True tensor ---
        im0 = axes[0].imshow(T_true, cmap="viridis", vmin=common_min, vmax=common_max)
        for (i, j), val in np.ndenumerate(T_true):
            axes[0].text(j, i, f"{val:.2e}", ha='center', va='center', color='white')
        axes[0].set_title(f"True (sample {idx})")
        fig.colorbar(im0, ax=axes[0], format="%.0e")

        # --- Predicted tensor ---
        im1 = axes[1].imshow(T_pred, cmap="viridis", vmin=common_min, vmax=common_max)
        for (i, j), val in np.ndenumerate(T_pred):
            axes[1].text(j, i, f"{val:.2e}", ha='center', va='center', color='white')
        axes[1].set_title(f"Predicted (sample {idx})")
        fig.colorbar(im1, ax=axes[1], format="%.0e")

        # --- Absolute Error tensor ---
        im2 = axes[2].imshow(T_err, cmap="RdBu_r", vmin=-err_max, vmax=err_max)
        for (i, j), val in np.ndenumerate(T_err):
            axes[2].text(j, i, f"{val:.2e}", ha='center', va='center', color='black')
        axes[2].set_title(f"Abs Error (True−Pred)")
        fig.colorbar(im2, ax=axes[2], format="%.0e")

        # --- Relative Error (% of True value) ---
        im3 = axes[3].imshow(T_relerr, cmap="inferno")
        for (i, j), val in np.ndenumerate(T_relerr):
            axes[3].text(j, i, f"{val:.2f}%", ha='center', va='center', color='white')
        axes[3].set_title("Relative Error (%)")
        fig.colorbar(im3, ax=axes[3], format="%.2f")

        plt.suptitle(f"{model_type} Stress Tensor Comparison (sample {idx})", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"stress_tensor_comparison_with_error_sample_{idx}.png"))
        plt.close()