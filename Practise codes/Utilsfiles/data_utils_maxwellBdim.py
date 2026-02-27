import os
import numpy as np
import torch
import jax.numpy as jnp
import flax
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import gaussian_kde

# =========================================================
# Dimensionless physics residual for Maxwell-B
# =========================================================
def maxwellB_residual_dimless(L_hat, T_hat, Wi):
    """
    Dimensionless form of steady-state Maxwell-B residual:
    T̂ − Wi (L̂^T T̂ + T̂ L̂) − 2 D̂ = 0
    """
    D_hat = 0.5 * (L_hat + jnp.swapaxes(L_hat, 1, 2))
    LTt   = jnp.matmul(jnp.swapaxes(L_hat, 1, 2), T_hat)
    TL    = jnp.matmul(T_hat, L_hat)
    return T_hat - Wi * (LTt + TL) - 2.0 * D_hat

# =========================================================
# Compute reference scales
# =========================================================
def compute_ref_scales(L_phys, T_phys_vec6, ETA0, LAM):
    """
    Compute scaling parameters from physical L and T data in vec6 format.

    Parameters
    ----------
    L_phys : ndarray, shape (N, 9)
        Flattened 3x3 velocity gradient tensor in physical units.
    T_phys_vec6 : ndarray, shape (N, 6)
        Symmetric stress tensor components in vec6 format: [xx, yy, zz, xy, xz, yz]
    ETA0 : float
        Zero-shear viscosity coefficient (physical units).
    LAM : float
        Relaxation time (physical units).

    Returns
    -------
    gamma_ref : float
        Median Frobenius norm of rate-of-deformation tensor D over training samples.
    sigma_ref_frob : float
        Median Frobenius norm of stress tensor T over training samples.
    sigma_vec : ndarray, shape (6,)
        Per-component standard deviation of T_phys_vec6 after Frobenius scaling.
    Wi : float
        Dimensionless Weissenberg number = LAM * gamma_ref.
    """
    # --- Step 1: convert L_phys (vec9) to full 3x3 tensor ---
    L = L_phys.reshape(-1, 3, 3)
    D = 0.5 * (L + np.swapaxes(L, 1, 2))  # symmetric part
    gamma_vals = np.linalg.norm(D, axis=(1, 2))
    gamma_ref = np.median(gamma_vals) + 1e-12

    # --- Step 2: helper to convert vec6 stresses to full symmetric 3x3 tensor ---
    def vec6_to_sym3_np(vec6):
        T = np.zeros((vec6.shape[0], 3, 3))
        T[:, 0, 0] = vec6[:, 0]  # xx
        T[:, 1, 1] = vec6[:, 1]  # yy
        T[:, 2, 2] = vec6[:, 2]  # zz
        T[:, 0, 1] = T[:, 1, 0] = vec6[:, 3]  # xy
        T[:, 0, 2] = T[:, 2, 0] = vec6[:, 4]  # xz
        T[:, 1, 2] = T[:, 2, 1] = vec6[:, 5]  # yz
        return T

    # --- Step 3: compute Frobenius norm scaling for T ---
    T_full = vec6_to_sym3_np(T_phys_vec6)
    sigma_ref_frob = np.median(np.linalg.norm(T_full, axis=(1, 2))) + 1e-12

    # --- Step 4: compute per-component scaling (from vec6 form) ---
    T_phys_scaled_frob = T_phys_vec6 / sigma_ref_frob
    sigma_vec = T_phys_scaled_frob.std(axis=0)
    sigma_vec[sigma_vec == 0] = 1.0  # safety: avoid div-by-zero

    # --- Step 5: compute Wi ---
    Wi = LAM * gamma_ref

    return gamma_ref, sigma_ref_frob, sigma_vec, Wi
# =========================================================
# Balanaced slpit
# =========================================================
def balanced_split_per_component(X, Y, test_size=0.1, val_size=0.2, seed=42):
    """
    Create train/val/test splits so each output component's distribution
    is balanced across the splits.

    Parameters
    ----------
    X : ndarray (N, 9)
        Flattened L tensors.
    Y : ndarray (N, 6)
        Stresses in vec6 format.
    test_size : float
        Fraction of data for test.
    val_size : float
        Fraction of data for val.
    seed : int
        Random seed.

    Returns
    -------
    X_train, X_val, X_test, Y_train, Y_val, Y_test
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    n_test_total = int(test_size * N)
    n_val_total = int(val_size * N)

    train_idx = set()
    val_idx = set()
    test_idx = set()

    # Number of samples to allocate per component
    per_comp_test = int(np.floor(n_test_total / Y.shape[1]))
    per_comp_val = int(np.floor(n_val_total / Y.shape[1]))

    for c in range(Y.shape[1]):
        # Sort indices by absolute magnitude for component c
        sorted_idx = np.argsort(np.abs(Y[:, c]))
        # Permute to avoid bias from ordering
        sorted_idx = sorted_idx[rng.permutation(len(sorted_idx))]

        # Allocate per component
        comp_test = sorted_idx[:per_comp_test]
        comp_val = sorted_idx[per_comp_test:per_comp_test + per_comp_val]
        comp_train = sorted_idx[per_comp_test + per_comp_val:]

        test_idx.update(comp_test)
        val_idx.update(comp_val)
        train_idx.update(comp_train)

    # Convert sets to arrays
    train_idx = np.array(list(train_idx))
    val_idx = np.array(list(val_idx))
    test_idx = np.array(list(test_idx))

    # Remove overlaps: train gets priority, then val, then test
    train_idx = np.setdiff1d(train_idx, np.union1d(val_idx, test_idx))
    val_idx = np.setdiff1d(val_idx, test_idx)

    # Final splits
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    return X_train, X_val, X_test, Y_train, Y_val, Y_test
# =========================================================
# Load and normalize data
# =========================================================
def load_and_normalize_data_maxwellB_dimless(
    X_path, Y_path, ETA0=5.28e-5, LAM=1.902,
    seed=42, test_size=0.1, val_size=0.2,
    balanced_split=False
):
    """
    Load Maxwell-B dataset, nondimensionalize L and T to O(1),
    and scale each stress component separately based on training set.
    Prints stats for physical and dimensionless data.
    """
    rng = np.random.default_rng(seed)

    # ===== Load data =====
    X = torch.load(X_path).numpy()
    Y = torch.load(Y_path).numpy()

    # ===== Train/val/test split =====
    if balanced_split:
        X_train, X_val, X_test, Y_train, Y_val, Y_test = balanced_split_per_component(
            X, Y, test_size=test_size, val_size=val_size, seed=seed
        )
    else:
        from sklearn.model_selection import train_test_split
        X_train_val, X_test, Y_train_val, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=seed, shuffle=True
        )
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_train_val, Y_train_val, test_size=val_size, random_state=seed
        )

    # ===== Debug helper =====
    def print_component_stats(name, arr):
        print(f"\n{name} shape={arr.shape}")
        for i in range(arr.shape[1]):
            comp = arr[:, i]
            print(f"  comp {i}: min={comp.min():.3e}, max={comp.max():.3e}, mean={comp.mean():.3e}, std={comp.std():.3e}")

    print("=== PHYSICAL L stats ===")
    print_component_stats("Train L", X_train)
    print_component_stats("Val L", X_val)
    print_component_stats("Test L", X_test)

    print("=== PHYSICAL T stats ===")
    print_component_stats("Train T", Y_train)
    print_component_stats("Val T", Y_val)
    print_component_stats("Test T", Y_test)

    # ===== Scaling parameters =====
    gamma_ref, sigma_ref_frob, sigma_vec, Wi = compute_ref_scales(X_train, Y_train, ETA0, LAM)

    print("\n=== Per-component sigma_vec (dimless unit) ===")
    for i, sv in enumerate(sigma_vec):
        print(f" comp {i}: {sv:.6e}")

    # ===== Make dimensionless =====
    def make_dimless(Xp, Yp):
        L_hat = Xp.reshape(-1, 3, 3) / gamma_ref
        T_scaled_frob = Yp / sigma_ref_frob
        T_hat = T_scaled_frob / sigma_vec
        return L_hat.reshape(Xp.shape[0], -1), T_hat

    X_train_hat, Y_train_hat = make_dimless(X_train, Y_train)
    X_val_hat, Y_val_hat = make_dimless(X_val, Y_val)
    X_test_hat, Y_test_hat = make_dimless(X_test, Y_test)

    # ===== Standardise using train stats =====
    Y_mean = Y_train_hat.mean(axis=0)
    Y_std = Y_train_hat.std(axis=0); Y_std[Y_std == 0] = 1.0
    Y_train_hat = (Y_train_hat - Y_mean) / Y_std
    Y_val_hat = (Y_val_hat - Y_mean) / Y_std
    Y_test_hat = (Y_test_hat - Y_mean) / Y_std

    # ===== Dimless stats =====
    print("\n=== DIMENSIONLESS L_hat stats ===")
    print_component_stats("Train L_hat", X_train_hat)
    print_component_stats("Val L_hat", X_val_hat)
    print_component_stats("Test L_hat", X_test_hat)

    print("=== DIMENSIONLESS T_hat stats ===")
    print_component_stats("Train T_hat", Y_train_hat)
    print_component_stats("Val T_hat", Y_val_hat)
    print_component_stats("Test T_hat", Y_test_hat)

    extras = {
        "gamma_ref": gamma_ref,
        "sigma_ref_frob": sigma_ref_frob,
        "sigma_vec": sigma_vec,
        "Wi": Wi
    }

    print("\n=== Dimensionless Scaling (MaxwellB) ===")
    print(f"gamma_ref: {gamma_ref:.6e}, sigma_ref_frob: {sigma_ref_frob:.6e}, Wi: {Wi:.6e}")

    return (
        jnp.array(X_train_hat), jnp.array(X_val_hat), jnp.array(X_test_hat),
        jnp.array(Y_train_hat), jnp.array(Y_val_hat), jnp.array(Y_test_hat),
        0.0, 1.0, Y_mean, Y_std, extras
    )

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
# Plotting Utilities (unchanged)
# =========================
def plot_learning_curves_physical(train_losses, val_losses, sigma_ref, fig_dir, model_type):
    """
    Plot training/validation loss curves in physical units for dimensionless scaling.
    Uses sigma_ref**2 to convert dimensionless MSE to physical MSE.
    """
    Y_var = float(sigma_ref**2)
    train_losses_phys = np.array(train_losses) * Y_var
    val_losses_phys   = np.array(val_losses)   * Y_var

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
    plt.ylabel(r"MSE Loss [Pa$^2$]", fontsize=14)
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
    eps = 1e-12
    for idx in sample_indices:
        T_true = np.array(vec6_to_sym3(jnp.array([y_true_phys[idx]]))).squeeze()
        T_pred = np.array(vec6_to_sym3(jnp.array([y_pred_phys[idx]]))).squeeze()
        T_err = T_true - T_pred
        T_relerr = np.abs(T_err) / (np.abs(T_true) + eps) * 100.0

        common_min = min(np.min(T_true), np.min(T_pred))
        common_max = max(np.max(T_true), np.max(T_pred))
        err_max = np.max(np.abs(T_err))

        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        im0 = axes[0].imshow(T_true, cmap="viridis", vmin=common_min, vmax=common_max)
        for (i, j), val in np.ndenumerate(T_true):
            axes[0].text(j, i, f"{val:.2e}", ha='center', va='center', color='white')
        axes[0].set_title(f"True (sample {idx})")
        fig.colorbar(im0, ax=axes[0], format="%.0e")

        im1 = axes[1].imshow(T_pred, cmap="viridis", vmin=common_min, vmax=common_max)
        for (i, j), val in np.ndenumerate(T_pred):
            axes[1].text(j, i, f"{val:.2e}", ha='center', va='center', color='white')
        axes[1].set_title(f"Predicted (sample {idx})")
        fig.colorbar(im1, ax=axes[1], format="%.0e")

        im2 = axes[2].imshow(T_err, cmap="RdBu_r", vmin=-err_max, vmax=err_max)
        for (i, j), val in np.ndenumerate(T_err):
            axes[2].text(j, i, f"{val:.2e}", ha='center', va='center', color='black')
        axes[2].set_title("Abs Error (True−Pred)")
        fig.colorbar(im2, ax=axes[2], format="%.0e")

        im3 = axes[3].imshow(T_relerr, cmap="inferno")
        for (i, j), val in np.ndenumerate(T_relerr):
            axes[3].text(j, i, f"{val:.2f}%", ha='center', va='center', color='white')
        axes[3].set_title("Relative Error (%)")
        fig.colorbar(im3, ax=axes[3], format="%.2f")

        plt.suptitle(f"{model_type} Stress Tensor Comparison (sample {idx})", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"stress_tensor_comparison_with_error_sample_{idx}.png"))
        plt.close()