import os
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
import jax.numpy as jnp
import flax
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import gaussian_kde

# ===========================================================
# 1. Stage-wise stable loader 
# ===========================================================
def load_and_normalize_stagewise_data_stable(
    model_type, data_root, stages,
    seed=42, scaling_mode="standard"
):
    """
    Loop through each stage folder and:
     - Load X/Y tensors
     - Split into train/val/test
     - Normalize using *stage training set stats*
     - Save normalized dataset stats + boxplots + histograms like your whole loader
     - Return arrays for later use in training
    """
    results = {}

    for stage_tag in stages:
        print(f"\n=== Stage {stage_tag} ({model_type}) ===")

        stage_dir = os.path.join(data_root, stage_tag)
        X_path = os.path.join(stage_dir, f"X_3D_{model_type}_stable.pt")
        Y_path = os.path.join(stage_dir, f"Y_3D_{model_type}_stable.pt")

        if not os.path.exists(X_path) or not os.path.exists(Y_path):
            print(f"[WARN] Missing files for stage {stage_tag}, skipping...")
            continue

        # Load data
        # ===== Load data =====
        print(f"\n📂 Loading stable dataset for {model_type}...")
        print(f"X file: {X_path}")
        print(f"Y file: {Y_path}")
        X = torch.load(X_path).numpy()
        Y = torch.load(Y_path).numpy()
        print(f"✅ Data loaded: X shape = {X.shape}, Y shape = {Y.shape}")
        

        # Split train/val/test
        X_train_val, X_test, Y_train_val, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=seed, shuffle=True
        )
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_train_val, Y_train_val, test_size=0.25, random_state=seed
        )
        print(f"📊 Split sizes: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")

        # Normalize X
        X_mean = X_train.mean(axis=0)
        X_std  = X_train.std(axis=0)
        X_std[X_std == 0] = 1.0
        X_train_n = (X_train - X_mean) / X_std
        X_val_n   = (X_val - X_mean) / X_std
        X_test_n  = (X_test - X_mean) / X_std

        # Normalize Y
        if scaling_mode == "standard":
            Y_mean = Y_train.mean(axis=0)
            Y_std  = Y_train.std(axis=0)
            Y_std[Y_std == 0] = 1.0
            Y_train_n = (Y_train - Y_mean) / Y_std
            Y_val_n   = (Y_val - Y_mean) / Y_std
            Y_test_n  = (Y_test - Y_mean) / Y_std
        elif scaling_mode == "minmax":
            Y_min = Y_train.min(axis=0)
            Y_max = Y_train.max(axis=0)
            Y_range = np.where((Y_max - Y_min) == 0, 1.0, Y_max - Y_min)
            Y_train_n = (Y_train - Y_min) / Y_range
            Y_val_n   = (Y_val - Y_min) / Y_range
            Y_test_n  = (Y_test - Y_min) / Y_range
            Y_mean, Y_std = Y_min, Y_range
        else:
            raise ValueError("scaling_mode must be 'standard' or 'minmax'")

        # Output directory
        norm_dir = os.path.join("images", f"{stage_tag} stable", model_type, "normalized")
        os.makedirs(norm_dir, exist_ok=True)

        # Write stats
        def write_stats(f, arr, label):
            flat = arr.flatten()
            mean_val = np.mean(flat)
            std_val  = np.std(flat)
            min_val  = np.min(flat)
            max_val  = np.max(flat)
            q1_val   = np.percentile(flat, 25)
            med_val  = np.percentile(flat, 50)
            q3_val   = np.percentile(flat, 75)
            iqr_val  = q3_val - q1_val
            count_val = flat.size

            f.write(f"--- {label} ---\n")
            f.write(f"Shape:         {arr.shape}\n")
            f.write(f"Count:         {count_val}\n")
            f.write(f"Mean:          {mean_val:.4e}\n")
            f.write(f"Std Dev:       {std_val:.4e}\n")
            f.write(f"Min:           {min_val:.4e}\n")
            f.write(f"25% (Q1):      {q1_val:.4e}\n")
            f.write(f"Median (Q2):   {med_val:.4e}\n")
            f.write(f"75% (Q3):      {q3_val:.4e}\n")
            f.write(f"Max:           {max_val:.4e}\n")
            f.write(f"IQR:           {iqr_val:.4e}\n")
            f.write(f"Range:         [{min_val:.4e}, {max_val:.4e}]\n\n")

        stats_path = os.path.join(norm_dir, "normalizeddatastat.txt")
        with open(stats_path, "w") as f:
            f.write(f"=== Normalized dataset stats for {model_type} Stage: {stage_tag} ===\n\n")
            write_stats(f, X_train_n, "Velocity Gradient (L) - Train Norm")
            write_stats(f, X_val_n,   "Velocity Gradient (L) - Val Norm")
            write_stats(f, X_test_n,  "Velocity Gradient (L) - Test Norm")
            write_stats(f, Y_train_n, "Stress Tensor (T) - Train Norm")
            write_stats(f, Y_val_n,   "Stress Tensor (T) - Val Norm")
            write_stats(f, Y_test_n,  "Stress Tensor (T) - Test Norm")

        # Plotting
        def plot_box_and_hist(X_data, Y_data, set_name):
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

        plot_box_and_hist(X_train_n, Y_train_n, "Train")
        plot_box_and_hist(X_val_n,   Y_val_n,   "Val")
        plot_box_and_hist(X_test_n,  Y_test_n,  "Test")

        print(f"✅ Normalized dataset stats & plots saved to: {norm_dir}")


        # Return arrays for this stage
        results[stage_tag] = (
            jnp.array(X_train_n), jnp.array(X_val_n), jnp.array(X_test_n),
            jnp.array(Y_train_n), jnp.array(Y_val_n), jnp.array(Y_test_n),
            X_mean, X_std, Y_mean, Y_std
        )

    return results

# ===========================================================
# 2. Save/load checkpoint utility
# ===========================================================
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

# ===============================================================
# 3. Plotting Utilities (identical to stable version, 6 functions)
# ===============================================================

def plot_all_losses(train_tot, val_tot,
                    train_d, val_d,
                    train_p, val_p,
                    Y_std, fig_dir, model_type):
    """
    Plot smoothed curves for:
      - Train/Val total loss
      - Train/Val data loss
      - Train/Val physics loss
    Loss is converted from normalized MSE to physical units [Pa^2]
    using the variance of Y_std.
    """
    Y_var = float(np.mean(np.array(Y_std) ** 2))
    #train_tot_ph = np.array(train_tot) * Y_var
    #val_tot_ph   = np.array(val_tot)   * Y_var
    train_d_ph   = np.array(train_d)   * Y_var
    val_d_ph     = np.array(val_d)     * Y_var
    train_p_ph   = np.array(train_p)   * Y_var
    val_p_ph     = np.array(val_p)     * Y_var

    def smooth_curve(values, weight=0.9):
        """Exponential smoothing for cleaner curves."""
        smoothed = []
        last = values[0]
        for val in values:
            last = last * weight + (1 - weight) * val
            smoothed.append(last)
        return smoothed

    #train_tot_ph = smooth_curve(train_tot_ph)
    #val_tot_ph   = smooth_curve(val_tot_ph)
    train_d_ph   = smooth_curve(train_d_ph)
    val_d_ph     = smooth_curve(val_d_ph)
    train_p_ph   = smooth_curve(train_p_ph)
    val_p_ph     = smooth_curve(val_p_ph)

    os.makedirs(fig_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    #plt.plot(train_tot_ph, label="Train Total Loss", color='tab:blue', linewidth=2)
    #plt.plot(val_tot_ph,   label="Val Total Loss",   color='tab:orange', linewidth=2)
    plt.plot(train_d_ph,   label="Train Data Loss",  color='tab:green', linestyle='-.')
    plt.plot(val_d_ph,     label="Val Data Loss",    color='tab:red', linestyle=':')
    plt.plot(train_p_ph,   label="Train Physics Loss", color='tab:purple', linestyle='-')
    plt.plot(val_p_ph,     label="Val Physics Loss", color='tab:brown', linestyle='--')
    plt.yscale("log")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel(r"MSE Loss [Pa$^2$]", fontsize=14)
    plt.title(f"All Losses ({model_type})", fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "all_losses_physical.png"), dpi=300)
    plt.close()

def plot_residual_hist(residuals, fig_dir, model_type):
    """
    Plot histogram of prediction residuals with KDE overlay and axis labels.
    """
    os.makedirs(fig_dir, exist_ok=True)
    residuals_1d = np.ravel(residuals)
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals_1d, bins=30, color='skyblue', stat='density')
    kde = gaussian_kde(residuals_1d)
    x_range = np.linspace(residuals_1d.min(), residuals_1d.max(), 1000)
    plt.plot(x_range, kde(x_range), color='orange', lw=2, label='KDE')
    plt.axvline(0, color='red', linestyle='--', label='Zero Residual')
    plt.xlabel("Residual")
    plt.ylabel("Density")
    plt.title(f"Residuals Histogram ({model_type})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "residual_histogram.png"))
    plt.close()

def plot_residuals_vs_pred(y_pred, residuals, fig_dir, model_type):
    """
    Scatter plot of residuals vs predicted values, with LOWESS smoothing.
    """
    os.makedirs(fig_dir, exist_ok=True)
    y_pred_1d = np.ravel(y_pred)
    residuals_1d = np.ravel(residuals)
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred_1d, residuals_1d, alpha=0.5)
    smoothed = sm.nonparametric.lowess(residuals_1d, y_pred_1d, frac=0.3)
    plt.plot(smoothed[:, 0], smoothed[:, 1], color='red', lw=2, label='LOWESS')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title(f"Residuals vs Predictions ({model_type})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "residuals_vs_predictions.png"))
    plt.close()

def plot_stress_tensor_comparison(vec6_to_sym3, y_true_phys, y_pred_phys, sample_indices, fig_dir, model_type):
    """
    Plot 4-panel stress tensor comparison in a 2×2 grid:
      [ True , Predicted ]
      [ Abs Error , Relative Error (%) ]
    Text color chosen for max contrast with background.
    """
    os.makedirs(fig_dir, exist_ok=True)
    eps = 1e-12

    def add_text_with_contrast(ax, matrix, cmap_name, fmt="{:.2e}"):
        norm = mpl.colors.Normalize(vmin=np.min(matrix), vmax=np.max(matrix))
        cmap = plt.get_cmap(cmap_name)
        for (i, j), val in np.ndenumerate(matrix):
            r, g, b, _ = cmap(norm(val))
            brightness = 0.299*r + 0.587*g + 0.114*b
            text_color = "black" if brightness > 0.6 else "white"
            ax.text(j, i, fmt.format(val), ha='center', va='center',
                    color=text_color, fontsize=9, fontweight='bold')

    for idx in sample_indices:
        T_true = np.array(vec6_to_sym3(jnp.array([y_true_phys[idx]]))).squeeze()
        T_pred = np.array(vec6_to_sym3(jnp.array([y_pred_phys[idx]]))).squeeze()
        T_err = T_true - T_pred
        T_relerr = np.abs(T_err) / (np.abs(T_true) + eps) * 100.0

        common_min = min(np.min(T_true), np.min(T_pred))
        common_max = max(np.max(T_true), np.max(T_pred))
        err_max = np.max(np.abs(T_err))

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        im0 = axes[0,0].imshow(T_true, cmap="viridis", vmin=common_min, vmax=common_max)
        add_text_with_contrast(axes[0,0], T_true, "viridis")
        axes[0,0].set_title(f"True (sample {idx})")
        fig.colorbar(im0, ax=axes[0,0], format="%.0e")

        im1 = axes[0,1].imshow(T_pred, cmap="viridis", vmin=common_min, vmax=common_max)
        add_text_with_contrast(axes[0,1], T_pred, "viridis")
        axes[0,1].set_title("Predicted")
        fig.colorbar(im1, ax=axes[0,1], format="%.0e")

        im2 = axes[1,0].imshow(T_err, cmap="RdBu_r", vmin=-err_max, vmax=err_max)
        add_text_with_contrast(axes[1,0], T_err, "RdBu_r")
        axes[1,0].set_title("Abs Error (True−Pred)")
        fig.colorbar(im2, ax=axes[1,0], format="%.0e")

        im3 = axes[1,1].imshow(T_relerr, cmap="inferno")
        add_text_with_contrast(axes[1,1], T_relerr, "inferno", fmt="{:.2f}%")
        axes[1,1].set_title("Relative Error (%)")
        fig.colorbar(im3, ax=axes[1,1], format="%.2f")

        plt.suptitle(f"{model_type} Stress Tensor Comparison (sample {idx})", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"stress_tensor_comparison_with_error_sample_{idx}.png"), dpi=300)
        plt.close()

def plot_dataset_predictions_summary(vec6_to_sym3, y_true_phys, y_pred_phys, fig_dir, model_type):
    """
    Aggregate dataset performance plots:
      - Abs/Rel error histograms side by side
      - True vs Predicted scatter with R²
      - Mean abs/rel error heatmaps per tensor component
    """
    os.makedirs(fig_dir, exist_ok=True)
    eps = 1e-12
    abs_err = np.abs(y_true_phys - y_pred_phys)
    rel_err = abs_err / (np.abs(y_true_phys) + eps) * 100

    # === Side-by-side histogram figure ===
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Absolute Error histogram
    ax[0].hist(abs_err.flatten(), bins=50, color='skyblue', edgecolor='black')
    ax[0].set_xlabel("Absolute Error")
    ax[0].set_ylabel("Frequency")
    ax[0].set_title(
        f"{model_type} - Abs Error\n"
        f"Mean={abs_err.mean():.2e}, Median={np.median(abs_err):.2e}"
    )
    ax[0].grid(True, ls="--")

    # Relative Error histogram (same style, zoomed to data range)
    ax[1].hist(rel_err.flatten(), bins=50, color='orange', edgecolor='black')
    ax[1].set_xlabel("Relative Error (%)")
    ax[1].set_ylabel("Frequency")
    ax[1].set_title(
        f"{model_type} - Rel Error\n"
        f"Mean={rel_err.mean():.2f}%, Median={np.median(rel_err):.2f}%"
    )
    rel_max = np.max(rel_err)
    ax[1].set_xlim(0, rel_max * 1.05)  # zoom to just above max val
    ax[1].grid(True, ls="--")

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "abs_rel_error_hist.png"), dpi=300)
    plt.close()

    # === Scatter plot: True vs Predicted ===
    plt.figure(figsize=(6,6))
    plt.scatter(y_true_phys.flatten(), y_pred_phys.flatten(), s=5, alpha=0.5)
    corr_coef = np.corrcoef(y_true_phys.flatten(), y_pred_phys.flatten())[0,1]
    plt.xlabel("True Stress Components")
    plt.ylabel("Predicted Stress Components")
    plt.title(f"{model_type} - True vs Predicted (R²={corr_coef**2:.3f})")
    plt.grid(True, ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "true_vs_pred_scatter.png"), dpi=300)
    plt.close()

    # --- Error heatmaps (absolute & relative) ---
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

    # Helper to annotate matrix cells with contrast-aware text
    def add_text_with_contrast(axh, matrix, cmap_name, fmt="{:.2e}"):
        norm = mpl.colors.Normalize(vmin=np.min(matrix), vmax=np.max(matrix))
        cmap = plt.get_cmap(cmap_name)
        for (i, j), val in np.ndenumerate(matrix):
            r, g, b, _ = cmap(norm(val))
            brightness = 0.299*r + 0.587*g + 0.114*b
            text_color = "black" if brightness > 0.6 else "white"
            axh.text(j, i, fmt.format(val), ha='center', va='center',
                     color=text_color, fontsize=9, fontweight='bold')

    # Mean Absolute Error Heatmap
    im0 = ax[0].imshow(mean_abs_err_per_tensor, cmap="RdBu_r")
    ax[0].set_title("Mean Absolute Error")
    fig.colorbar(im0, ax=ax[0])
    add_text_with_contrast(ax[0], mean_abs_err_per_tensor, "RdBu_r", fmt="{:.2e}")
    ax[0].set_xticks([])  # remove tick coords
    ax[0].set_yticks([])

    # Mean Relative Error Heatmap
    im1 = ax[1].imshow(mean_rel_err_per_tensor, cmap="inferno")
    ax[1].set_title("Mean Relative Error (%)")
    fig.colorbar(im1, ax=ax[1])
    add_text_with_contrast(ax[1], mean_rel_err_per_tensor, "inferno", fmt="{:.2f}%")
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    plt.suptitle(f"{model_type} - Dataset-level Error Heatmaps", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "mean_error_heatmaps.png"), dpi=300)
    plt.close()
      
#=========================================================================  
# 4. To run just the stage-wise loader by itself  
#=========================================================================  
@hydra.main(config_path="../config/train", config_name="stable_tensor_config", version_base=None)
def main(cfg: DictConfig):
    model_type = cfg.model_type

    # Stages to process
    stages_list = [
        "1.0", "1.0_1.2", "1.2_1.4", "1.4_1.6", "1.6_1.8",
        "1.8_2.0", "2.0_2.2", "2.2_2.4", "2.4_2.6", "2.6_2.8"
    ]

    # Root folder containing stage subfolders
    data_root = "datafiles"

    print(f"\n=== Processing stage-wise normalization in {data_root} for {model_type} ===")
    results = load_and_normalize_stagewise_data_stable(
        model_type=model_type,
        data_root=data_root,
        stages=stages_list,
        seed=cfg.seed,
        scaling_mode=cfg.data.scaling_mode
    )

    # Optional: show processed stages
    print("✅ Processed stages:", list(results.keys()))
    
    # Now `results` is a dictionary keyed by stage_tag, with normalized splits + params
    return results
    

if __name__ == "__main__":
    main()

   