import os
import torch
import jax.numpy as jnp
import numpy as np
import flax
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import statsmodels.api as sm
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =============================
# Balanced split helper
# =============================
def balanced_split_quantiles(X, Y, train_frac=0.6, val_frac=0.2, test_frac=0.2,
                             n_bins=10, seed=42):
    """
    Stratified balanced split based on Frobenius norm magnitude using quantile bins.
    Works for both Maxwell-B and Oldroyd-B data.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-8, "Fractions must sum to 1."

    magnitudes = np.sqrt(
        Y[:, 0]**2 + Y[:, 1]**2 + Y[:, 2]**2 +
        2*(Y[:, 3]**2 + Y[:, 4]**2 + Y[:, 5]**2)
    )

    rng = np.random.default_rng(seed)
    quantile_edges = np.quantile(magnitudes, np.linspace(0, 1, n_bins+1))

    train_idx, val_idx, test_idx = [], [], []

    for i in range(n_bins):
        bin_mask = (magnitudes >= quantile_edges[i]) & (magnitudes <= quantile_edges[i+1])
        bin_indices = np.where(bin_mask)[0]
        rng.shuffle(bin_indices)

        n_bin = len(bin_indices)
        n_train = int(round(train_frac * n_bin))
        n_val   = int(round(val_frac * n_bin))
        n_test  = n_bin - n_train - n_val

        train_idx.extend(bin_indices[:n_train])
        val_idx.extend(bin_indices[n_train:n_train+n_val])
        test_idx.extend(bin_indices[n_train+n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return (X[train_idx], Y[train_idx],
            X[val_idx], Y[val_idx],
            X[test_idx], Y[test_idx])

def signed_log_transform(T):
    import numpy as np
    return np.sign(T) * np.log(np.abs(T) + 1.0)

def signed_log_inverse(T_prime):
    import numpy as np
    return np.sign(T_prime) * (np.exp(np.abs(T_prime)) - 1.0)

# =============================
# Unified loader
# =============================
def load_and_normalize_data_unstable(
    model_type,                   # "maxwell_B" or "oldroyd_B"
    X_path, Y_path,
    seed=42, test_size=0.2, val_size=0.2,
    balanced_split=False, scaling_mode="standard"
):
    """
    Loads unstable Maxwell-B or Oldroyd-B data, splits into train/val/test, normalizes.
    Also saves normalized dataset stats + boxplots + histograms in images/unstable/<model_type>/normalized
    """

    rng = np.random.default_rng(seed)

    # ===== Load data =====
    print(f"\n📂 Loading unstable dataset for {model_type}...")
    print(f"X file: {X_path}")
    print(f"Y file: {Y_path}")
    X = torch.load(X_path).numpy()
    Y = torch.load(Y_path).numpy()
    print(f"✅ Data loaded: X shape = {X.shape}, Y shape = {Y.shape}")

    # ===== Split =====
    if balanced_split:
        X_train, Y_train, X_val, Y_val, X_test, Y_test = \
            balanced_split_quantiles(X, Y, test_frac=test_size, val_frac=val_size, n_bins=10, seed=seed)
    else:
        
        X_train_val, X_test, Y_train_val, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=seed, shuffle=True
        )
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_train_val, Y_train_val, test_size=val_size/(1-test_size), random_state=seed
        )
    print(f"📊 Split sizes: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
    # ===== Normalize X =====
    X_mean = X_train.mean(axis=0)
    X_std  = X_train.std(axis=0)
    X_std[X_std == 0] = 1.0

    X_train_n = (X_train - X_mean) / X_std
    X_val_n   = (X_val   - X_mean) / X_std
    X_test_n  = (X_test  - X_mean) / X_std

    # ===== Normalize/Scale Y =====
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

        Y_mean, Y_std = Y_min, Y_range
    else:
        raise ValueError("scaling_mode must be 'standard' or 'minmax'")

    # === Normalized folder path ===
    norm_dir = os.path.join("images", "unstable", model_type, "normalized")
    os.makedirs(norm_dir, exist_ok=True)

    # === Helper: write stats for an array ===
    def write_stats(f, arr, label):
        flat = arr.flatten()
        mean_val = np.mean(flat)
        std_val = np.std(flat)
        min_val = np.min(flat)
        max_val = np.max(flat)
        q1_val = np.percentile(flat, 25)
        med_val = np.percentile(flat, 50)
        q3_val = np.percentile(flat, 75)
        iqr_val = q3_val - q1_val
        count_val = flat.size
        f.write(f"--- {label} ---\n")
        f.write(f"  Shape:      {arr.shape}\n")
        f.write(f"  Count:      {count_val}\n")
        f.write(f"  Mean:       {mean_val:.4e}\n")
        f.write(f"  Std Dev:    {std_val:.4e}\n")
        f.write(f"  Min:        {min_val:.4e}\n")
        f.write(f"  25% (Q1):   {q1_val:.4e}\n")
        f.write(f"  Median(Q2): {med_val:.4e}\n")
        f.write(f"  75% (Q3):   {q3_val:.4e}\n")
        f.write(f"  IQR:        {iqr_val:.4e}\n")
        f.write(f"  Max:        {max_val:.4e}\n")
        f.write(f"  Range:      [{min_val:.4e}, {max_val:.4e}]\n\n")

    # === Write normalizeddatastat.txt ===
    stats_path = os.path.join(norm_dir, "normalizeddatastat.txt")
    with open(stats_path, "w") as f:
        f.write(f"=== Normalized dataset stats for {model_type} (Unstable) ===\n\n")
        write_stats(f, X_train_n, "Velocity Gradient (L) - Train Norm")
        write_stats(f, X_val_n,   "Velocity Gradient (L) - Val Norm")
        write_stats(f, X_test_n,  "Velocity Gradient (L) - Test Norm")
        write_stats(f, Y_train_n, "Stress Tensor (T) - Train Norm")
        write_stats(f, Y_val_n,   "Stress Tensor (T) - Val Norm")
        write_stats(f, Y_test_n,  "Stress Tensor (T) - Test Norm")

    # === Helper: plot boxplot & histogram in same figure ===
    def plot_box_and_hist(X_data, Y_data, set_name):
        # Boxplot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].boxplot(X_data.flatten(), vert=True, patch_artist=True, boxprops=dict(facecolor="lightblue"))
        axes[0].set_title(f"Boxplot - L ({set_name})")
        axes[0].grid(True, ls="--", alpha=0.5)
        axes[1].boxplot(Y_data.flatten(), vert=True, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
        axes[1].set_title(f"Boxplot - T ({set_name})")
        axes[1].grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(norm_dir, f"boxplots_XY_{set_name.lower()}.png"), dpi=300)
        plt.close()

        # Histogram
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].hist(X_data.flatten(), bins=50, color='steelblue', edgecolor='white', log=True)
        axes[0].set_title(f"Histogram - L ({set_name})")
        axes[0].grid(True, ls="--", alpha=0.5)
        axes[1].hist(Y_data.flatten(), bins=50, color='orange', edgecolor='white', log=True)
        axes[1].set_title(f"Histogram - T ({set_name})")
        axes[1].grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(norm_dir, f"histograms_XY_{set_name.lower()}.png"), dpi=300)
        plt.close()

    # === Generate plots for Train, Val, Test ===
    plot_box_and_hist(X_train_n, Y_train_n, "Train")
    plot_box_and_hist(X_val_n, Y_val_n, "Val")
    plot_box_and_hist(X_test_n, Y_test_n, "Test")

    print(f"✅ Normalized dataset stats & plots saved to: {norm_dir}")

    return (jnp.array(X_train_n), jnp.array(X_val_n), jnp.array(X_test_n),
            jnp.array(Y_train_n), jnp.array(Y_val_n), jnp.array(Y_test_n),
            X_mean, X_std, Y_mean, Y_std)

# ============================
# Checkpoint utils (unchanged)
# =============================
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
def plot_all_losses(train_tot, val_tot,
                    train_d, val_d,
                    train_p, val_p,
                    Y_std, fig_dir, model_type):
    """
    Plot total, data, and physics loss curves for both
    training and validation sets, converted to physical units.
    """

    # Convert normalized MSE to physical MSE using mean variance of Y_std
    Y_var = float(np.mean(np.array(Y_std)**2))
    #train_tot_ph = np.array(train_tot) * Y_var
    #val_tot_ph   = np.array(val_tot) * Y_var
    train_d_ph   = np.array(train_d) * Y_var
    val_d_ph     = np.array(val_d) * Y_var
    train_p_ph   = np.array(train_p) * Y_var
    val_p_ph     = np.array(val_p) * Y_var

    # Smoothing helper
    def smooth_curve(values, weight=0.9):
        smoothed = []
        last = values[0]
        for val in values:
            smoothed_val = last * weight + (1 - weight) * val
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    # Smooth all six curves
    #train_tot_ph = smooth_curve(train_tot_ph)
    #val_tot_ph   = smooth_curve(val_tot_ph)
    train_d_ph   = smooth_curve(train_d_ph)
    val_d_ph     = smooth_curve(val_d_ph)
    train_p_ph   = smooth_curve(train_p_ph)
    val_p_ph     = smooth_curve(val_p_ph)

    # Plot
    plt.figure(figsize=(10, 6))
    #plt.plot(train_tot_ph,  label="Train Total Loss",   color='tab:blue',   linewidth=2)
    #plt.plot(val_tot_ph,    label="Val Total Loss",     color='tab:orange', linewidth=2)
    plt.plot(train_d_ph,    label="Train Data Loss",    color='tab:green',  linestyle='-.', marker='^', markersize=4)
    plt.plot(val_d_ph,      label="Val Data Loss",      color='tab:red',    linestyle=':',  marker='d', markersize=4)
    plt.plot(train_p_ph,    label="Train Physics Loss", color='tab:purple', linestyle='-',  marker='x', markersize=4)
    plt.plot(val_p_ph,      label="Val Physics Loss",   color='tab:brown',  linestyle='--', marker='*', markersize=4)

    plt.yscale("log")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel(r"MSE Loss [Pa$^2$]", fontsize=14)
    plt.title(f"All Losses ({model_type})", fontsize=16)
    plt.legend(fontsize=10, loc="upper right")
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.tight_layout()

    save_path = os.path.join(fig_dir, "all_losses_physical.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Saved all-losses plot to {save_path}")

    
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
    """
    Plot 4-panel stress tensor comparison in a 2×2 grid:
      [ True , Predicted ]
      [ Abs Error , Relative Error (%) ]
    Text colour is automatically chosen per-cell for max contrast with background colour.
    """
    os.makedirs(save_dir, exist_ok=True)
    eps = 1e-12  # avoid division-by-zero in relative error

    def add_text_with_contrast(ax, matrix, cmap_name, fmt="{:.2e}"):
        """Add text to each cell with colour chosen for max contrast."""
        norm = mpl.colors.Normalize(vmin=np.min(matrix), vmax=np.max(matrix))
        cmap = plt.get_cmap(cmap_name)
        for (i, j), val in np.ndenumerate(matrix):
            r, g, b, _ = cmap(norm(val))
            brightness = 0.299*r + 0.587*g + 0.114*b  # luminance
            text_color = "black" if brightness > 0.6 else "white"
            ax.text(j, i, fmt.format(val),
                    ha='center', va='center',
                    color=text_color, fontsize=9, fontweight='bold')

    for idx in sample_indices:
        # Convert vector → symmetric 3x3 tensors
        T_true = np.array(vec6_to_sym3(jnp.array([y_true_phys[idx]]))).squeeze()
        T_pred = np.array(vec6_to_sym3(jnp.array([y_pred_phys[idx]]))).squeeze()
        T_err = T_true - T_pred
        T_relerr = np.abs(T_err) / (np.abs(T_true) + eps) * 100.0  # % relative error

        # Use same colour scale for True & Pred
        common_min = min(np.min(T_true), np.min(T_pred))
        common_max = max(np.max(T_true), np.max(T_pred))
        err_max = np.max(np.abs(T_err))

        # Create 2x2 grid figure
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        # --- True tensor ---
        im0 = axes[0, 0].imshow(T_true, cmap="viridis", vmin=common_min, vmax=common_max)
        add_text_with_contrast(axes[0, 0], T_true, "viridis", fmt="{:.2e}")
        axes[0, 0].set_title(f"True (sample {idx})")
        axes[0, 0].set_xticks([])
        axes[0, 0].set_yticks([])   # remove y-axis ticks
        fig.colorbar(im0, ax=axes[0, 0], format="%.0e")

        # --- Predicted tensor ---
        im1 = axes[0, 1].imshow(T_pred, cmap="viridis", vmin=common_min, vmax=common_max)
        add_text_with_contrast(axes[0, 1], T_pred, "viridis", fmt="{:.2e}")
        axes[0, 1].set_title(f"Predicted (sample {idx})")
        axes[0, 0].set_xticks([])
        axes[0, 0].set_yticks([])   # remove y-axis ticks
        fig.colorbar(im1, ax=axes[0, 1], format="%.0e")

        # --- Absolute Error tensor ---
        im2 = axes[1, 0].imshow(T_err, cmap="RdBu_r", vmin=-err_max, vmax=err_max)
        add_text_with_contrast(axes[1, 0], T_err, "RdBu_r", fmt="{:.2e}")
        axes[1, 0].set_title("Abs Error (True−Pred)")
        axes[0, 0].set_xticks([])
        axes[0, 0].set_yticks([])   # remove y-axis ticks
        fig.colorbar(im2, ax=axes[1, 0], format="%.0e")

        # --- Relative Error (%) ---
        im3 = axes[1, 1].imshow(T_relerr, cmap="inferno")
        add_text_with_contrast(axes[1, 1], T_relerr, "inferno", fmt="{:.2f}%")
        axes[1, 1].set_title("Relative Error (%)")
        axes[0, 0].set_xticks([])
        axes[0, 0].set_yticks([])   # remove y-axis ticks
        fig.colorbar(im3, ax=axes[1, 1], format="%.2f")

        # Final layout and save
        plt.suptitle(f"{model_type} Stress Tensor Comparison (sample {idx})", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,
                                 f"stress_tensor_comparison_with_error_sample_{idx}.png"),
                    dpi=300)
        plt.close()

def compare_test_set_T_and_save(vec6_to_sym3, y_true_phys, y_pred_phys, fig_dir, model_type, metrics_table=None):
    """
    Compare predicted vs ground truth stress tensors for the full test set.
    Saves:
      - Component metrics table (MSE, RMSE, R², MeanAbsErr, MeanRelErr%)
      - One scatter plot: True vs Predicted (all components)
      - One side-by-side abs-error hist and rel-error hist (all components)
      - Error heatmaps (absolute & relative)
      - Overall metrics table at the end (optional)
    """
    os.makedirs(fig_dir, exist_ok=True)
    labels = ["σ_xx", "σ_yy", "σ_zz", "σ_xy", "σ_xz", "σ_yz"]
    eps = 1e-12

    # Output file path
    metrics_path = os.path.join(fig_dir, f"{model_type}_testset_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        # === Write header ===
        f.write(f"=== Test Set Metrics Summary for {model_type} ===\n\n")
        header = "{:<10} {:>12} {:>12} {:>8} {:>15} {:>18} {:>18}\n".format(
            "Component", "MSE", "RMSE", "R²", "MeanAbsErr", "MeanRelErr(%)", "N_Samples"
        )
        f.write(header)
        f.write("-" * len(header) + "\n")

        # === Loop through each component ===
        for k in range(6):
            mse = mean_squared_error(y_true_phys[:, k], y_pred_phys[:, k])
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true_phys[:, k], y_pred_phys[:, k])
            abs_err_comp = np.abs(y_true_phys[:, k] - y_pred_phys[:, k])
            mean_abs_err_val = np.mean(abs_err_comp)
            rel_err_comp = abs_err_comp / (np.abs(y_true_phys[:, k]) + eps) * 100.0
            mean_rel_err_val = np.mean(rel_err_comp)
            n_samples = y_true_phys.shape[0]

            # Save to file
            f.write("{:<10} {:>12.4e} {:>12.4e} {:>8.4f} {:>15.4e} {:>18.2f} {:>18}\n".format(
                labels[k], mse, rmse, r2, mean_abs_err_val, mean_rel_err_val, n_samples
            ))

        # === Write overall metrics table once ===
        if metrics_table is not None:
            f.write("\n=== Overall Metrics Summary ===\n\n")
            header_overall = "{:<25} {:>15} {:>15}\n".format("Metric", "Value", "RMSE [Pa] if MSE")
            f.write(header_overall)
            f.write("-" * len(header_overall) + "\n")
            for metric, value in metrics_table:
                if "MSE" in metric or "loss" in metric.lower():
                    rmse_val = np.sqrt(value) if value >= 0 else np.nan
                    f.write("{:<25} {:>15.6f} {:>15.6f}\n".format(metric, value, rmse_val))
                else:
                    f.write("{:<25} {:>15.6f} {:>15}\n".format(metric, value, ""))

    # ====== Whole-test-set plots ======

    # Flatten all components
    y_true_all = y_true_phys.flatten()
    y_pred_all = y_pred_phys.flatten()

    # Absolute and relative error arrays
    abs_err_all = np.abs(y_true_all - y_pred_all)
    rel_err_all = abs_err_all / (np.abs(y_true_all) + eps) * 100.0

    # === Side-by-side histograms ===
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].hist(abs_err_all, bins=50, color='skyblue', edgecolor='black')
    ax[0].set_xlabel("Absolute Error [Pa]")
    ax[0].set_ylabel("Frequency")
    ax[0].set_title(
        f"{model_type} - Abs Error\n"
        f"Mean={abs_err_all.mean():.2e}, Median={np.median(abs_err_all):.2e}"
    )
    ax[0].grid(True, ls="--")

    ax[1].hist(rel_err_all, bins=50, color='orange', edgecolor='black')
    ax[1].set_xlabel("Relative Error (%)")
    ax[1].set_ylabel("Frequency")
    ax[1].set_title(
        f"{model_type} - Rel Error\n"
        f"Mean={rel_err_all.mean():.2f}%, Median={np.median(rel_err_all):.2f}%"
    )
    rel_max = np.max(rel_err_all)
    ax[1].set_xlim(0, rel_max * 1.05)
    ax[1].grid(True, ls="--")

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "abs_rel_error_hist_testset.png"), dpi=300)
    plt.close()

    # === Scatter plot: True vs Predicted ===
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_all, y_pred_all, s=5, alpha=0.5)
    corr_coef = np.corrcoef(y_true_all, y_pred_all)[0, 1]
    plt.xlabel("True Stress Components [Pa]")
    plt.ylabel("Predicted Stress Components [Pa]")
    plt.title(f"{model_type} - True vs Predicted (R²={corr_coef**2:.3f})")
    plt.grid(True, ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "true_vs_pred_scatter_testset.png"), dpi=300)
    plt.close()

    # === Error heatmaps ===
    mean_abs_err_per_tensor = np.mean(
        np.abs(vec6_to_sym3(np.array(y_true_phys)) - vec6_to_sym3(np.array(y_pred_phys))),
        axis=0
    )
    mean_rel_err_per_tensor = np.mean(
        (np.abs(vec6_to_sym3(np.array(y_true_phys)) - vec6_to_sym3(np.array(y_pred_phys))) /
         (np.abs(vec6_to_sym3(np.array(y_true_phys))) + eps) * 100),
        axis=0
    )

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    def add_text_with_contrast(axh, matrix, cmap_name, fmt="{:.2e}"):
        norm = mpl.colors.Normalize(vmin=np.min(matrix), vmax=np.max(matrix))
        cmap = plt.get_cmap(cmap_name)
        for (i, j), val in np.ndenumerate(matrix):
            r, g, b, _ = cmap(norm(val))
            brightness = 0.299*r + 0.587*g + 0.114*b
            text_color = "black" if brightness > 0.6 else "white"
            axh.text(j, i, fmt.format(val), ha='center', va='center',
                     color=text_color, fontsize=9, fontweight='bold')

    im0 = ax[0].imshow(mean_abs_err_per_tensor, cmap="RdBu_r")
    ax[0].set_title("Mean Absolute Error")
    fig.colorbar(im0, ax=ax[0])
    add_text_with_contrast(ax[0], mean_abs_err_per_tensor, "RdBu_r", fmt="{:.2e}")
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    im1 = ax[1].imshow(mean_rel_err_per_tensor, cmap="inferno")
    ax[1].set_title("Mean Relative Error (%)")
    fig.colorbar(im1, ax=ax[1])
    add_text_with_contrast(ax[1], mean_rel_err_per_tensor, "inferno", fmt="{:.2f}%")
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    plt.suptitle(f"{model_type} - Dataset-level Error Heatmaps (Test Set)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "mean_error_heatmaps_testset.png"), dpi=300)
    plt.close()

    print(f"✅ Test set metrics saved to: {metrics_path}")
    print(f"✅ Combined test set plots saved in: {fig_dir}")
        
#=========================================================================
# To run just the loader by itself
#=========================================================================
@hydra.main(config_path="../config/train", config_name="unstable_tensor_config", version_base=None)
def main(cfg: DictConfig):
    # Pull model_type from config
    model_type = cfg.model_type
    X_path = cfg.data.paths[model_type].x
    Y_path = cfg.data.paths[model_type].y
    
    print(f"\n=== Processing for {model_type} ===")
    load_and_normalize_data_unstable(
        model_type,
        X_path,
        Y_path,
        seed=cfg.seed,
        scaling_mode=cfg.data.scaling_mode
    )

if __name__ == "__main__":
    main()

      