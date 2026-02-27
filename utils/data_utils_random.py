#!/usr/bin/env python3
"""
Standalone Stage-wise Replay Loader + Utilities
"""

import os
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
import jax.numpy as jnp
import flax
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import statsmodels.api as sm
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error, r2_score

# ===========================================================
# 1. Stage-wise Replay Loader (UPDATED)
# ===========================================================
def load_and_normalize_stagewise_data_replay(
    model_type, data_root, mode, seed,  # <--- Added 'mode' and 'seed' as explicit args
    scaling_mode="standard", replay_ratio=0.2
):
    """
    Stage-wise loader with experience replay.
    Adapts to 'single_stage' or 'multi_stage' folder structures.
    """
    results = {}
    
    # 1. Define Stage Order based on Mode (Matching generateRandomdata.py)
    if mode == "single_stage":
        stage_order = ["1.0_2.4"]
    elif mode == "multi_stage":
        # Note: '1.0' is merged into '1.0_1.2' in your generator
        stage_order = [
            "1.0_1.2", "1.2_1.4", "1.4_1.6", "1.6_1.8", 
            "1.8_2.0", "2.0_2.2", "2.2_2.4"
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 2. Iterate over stages
    for stage_tag in stage_order:
        print(f"\n=== Stage {stage_tag} ({model_type}) - REPLAY MODE (Ratio={replay_ratio}) ===")
        
        # 3. Construct Path (Matching: datafiles/random/mode/seed/stage)
        # We assume data_root is "datafiles"
        stage_dir = os.path.join(data_root, "random", mode, f"seed_{seed}", stage_tag)
        
        # 4. Construct Filename (Matching: X_3D_{model}_stage.pt)
        # Note suffix is '_stage.pt' not '_stable.pt'
        X_path = os.path.join(stage_dir, f"X_3D_{model_type}_stage.pt")
        Y_path = os.path.join(stage_dir, f"Y_3D_{model_type}_stage.pt")

        if not (os.path.exists(X_path) and os.path.exists(Y_path)):
            print(f"[WARN] Missing files for stage {stage_tag} at {stage_dir}, skipping...")
            continue

        # Load current stage data
        X = torch.load(X_path)
        Y = torch.load(Y_path)

        # 5. Experience Replay (Only for Multi-Stage > index 0)
        # We skip replay if it's the first stage OR if we are in single_stage mode
        if mode == "multi_stage" and stage_tag in stage_order and stage_order.index(stage_tag) > 0 and replay_ratio > 0:
            current_idx = stage_order.index(stage_tag)
            num_current_samples = X.shape[0]
            num_replay = int(num_current_samples * replay_ratio)

            print(f"   ↺ Adding {num_replay} replay samples from previous stages...")
            replay_X_list, replay_Y_list = [], []
            
            # Gather data from ALL previous stages
            for prev_stage in stage_order[:current_idx]:
                prev_dir = os.path.join(data_root, "random", mode, f"seed_{seed}", prev_stage)
                pX_path = os.path.join(prev_dir, f"X_3D_{model_type}_stage.pt")
                pY_path = os.path.join(prev_dir, f"Y_3D_{model_type}_stage.pt")
                
                if os.path.exists(pX_path) and os.path.exists(pY_path):
                    replay_X_list.append(torch.load(pX_path))
                    replay_Y_list.append(torch.load(pY_path))

            if replay_X_list:
                all_prev_X = torch.cat(replay_X_list, dim=0)
                all_prev_Y = torch.cat(replay_Y_list, dim=0)
                
                # Randomly sample the required amount
                if all_prev_X.shape[0] >= num_replay:
                    idxs = torch.randperm(all_prev_X.shape[0])[:num_replay]
                    X_replay = all_prev_X[idxs]
                    Y_replay = all_prev_Y[idxs]
                    
                    X = torch.cat([X, X_replay], dim=0)
                    Y = torch.cat([Y, Y_replay], dim=0)
                    print(f"   ✅ Combined samples: {num_current_samples} + {X_replay.shape[0]} replay")
                else:
                    print(f"   ⚠️ Not enough replay samples available, using all {all_prev_X.shape[0]}")
                    X = torch.cat([X, all_prev_X], dim=0)
                    Y = torch.cat([Y, all_prev_Y], dim=0)

        # Convert to NumPy
        X = X.numpy()
        Y = Y.numpy()
        print(f"Final shapes: X={X.shape}, Y={Y.shape}")

        # Split train/val/test
        # We use a consistent random_state=seed to ensure splits are reproducible
        X_train_val, X_test, Y_train_val, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=seed, shuffle=True
        )
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_train_val, Y_train_val, test_size=0.25, random_state=seed
        )

        # Normalize X (Standard Scaler)
        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0)
        X_std[X_std == 0] = 1.0 # Prevent div by zero
        
        X_train_n = (X_train - X_mean) / X_std
        X_val_n = (X_val - X_mean) / X_std
        X_test_n = (X_test - X_mean) / X_std

        # Normalize Y
        if scaling_mode == "standard":
            Y_mean = Y_train.mean(axis=0)
            Y_std = Y_train.std(axis=0)
            Y_std[Y_std == 0] = 1.0
            
            Y_train_n = (Y_train - Y_mean) / Y_std
            Y_val_n = (Y_val - Y_mean) / Y_std
            Y_test_n = (Y_test - Y_mean) / Y_std
            
        elif scaling_mode == "minmax":
            Y_min = Y_train.min(axis=0)
            Y_max = Y_train.max(axis=0)
            Y_range = np.where((Y_max - Y_min) == 0, 1.0, Y_max - Y_min)
            
            Y_train_n = (Y_train - Y_min) / Y_range
            Y_val_n = (Y_val - Y_min) / Y_range
            Y_test_n = (Y_test - Y_min) / Y_range
            Y_mean, Y_std = Y_min, Y_range
            
        else:
            raise ValueError(f"Invalid scaling_mode: {scaling_mode}")

        # =========================================================
        # SAVE STATS & PLOTS (Updated Block)
        # =========================================================
        norm_dir = os.path.join("images", "random", mode, f"seed_{seed}", f"{stage_tag}_replay_analysis")
        os.makedirs(norm_dir, exist_ok=True)

        # 1. Write Detailed Stats (Your improved version)
        def write_stats(f, arr, label):
            flat = arr.flatten()
            mean_val, std_val = np.mean(arr), np.std(arr)
            min_val, max_val = np.min(arr), np.max(arr)
            q1_val, med_val, q3_val = np.percentile(arr, [25, 50, 75])
            iqr_val = q3_val - q1_val
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

        with open(os.path.join(norm_dir, "normalizeddatastat.txt"), "w") as f:
            f.write(f"=== Stats for {model_type} Stage {stage_tag} (Replay) ===\n")
            write_stats(f, X_train_n, "Velocity Gradient (L) - Train Norm")
            write_stats(f, X_val_n,   "Velocity Gradient (L) - Val Norm")
            write_stats(f, X_test_n,  "Velocity Gradient (L) - Test Norm")
            write_stats(f, Y_train_n, "Stress Tensor (T) - Train Norm")
            write_stats(f, Y_val_n,   "Stress Tensor (T) - Val Norm")
            write_stats(f, Y_test_n,  "Stress Tensor (T) - Test Norm")

        # 2. Plot Improved Histograms (Matches write_sampledata1.py style)
        def plot_hist_improved(X_data, Y_data, set_name):
            # Style settings
            plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

            # Clean model name for display
            clean_model_name = model_type.replace("_", " ").title()

            # Helper: Stats Box
            def add_stats_box(ax, data, color):
                mean_val, std_val = np.mean(data), np.std(data)
                text_str = '\n'.join((r'$\mu=%.3f$' % mean_val, r'$\sigma=%.3f$' % std_val))
                props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=color)
                ax.text(0.95, 0.95, text_str, transform=ax.transAxes, fontsize=11,
                        verticalalignment='top', horizontalalignment='right', bbox=props)

            # Helper: Info Box
            def add_info_box(ax, set_label):
                text_str = f"Model: {clean_model_name}\nStage: {stage_tag}\nSet: {set_label}"
                props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
                ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', horizontalalignment='left', bbox=props)

            # --- Plot L (SteelBlue) ---
            axes[0].hist(X_data.flatten(), bins=60, color='#4682B4', 
                         edgecolor='black', linewidth=0.5, log=True, alpha=1.0)
            axes[0].set_title(fr"Normalized Distribution of $\mathbf{{L}}$ ({set_name})", fontsize=14, pad=10)
            axes[0].set_xlabel(r"Normalized Value ($z$-score)", fontsize=14)
            axes[0].set_ylabel("Frequency (Log Scale)", fontsize=14)
            axes[0].grid(True, which="both", ls="--", alpha=0.3)
            add_stats_box(axes[0], X_data.flatten(), '#4682B4')
            add_info_box(axes[0], set_name)

            # --- Plot T (DarkOrange) ---
            axes[1].hist(Y_data.flatten(), bins=60, color='#FF8C00', 
                         edgecolor='black', linewidth=0.5, log=True, alpha=1.0)
            axes[1].set_title(fr"Normalized Distribution of $\mathbf{{T}}$ ({set_name})", fontsize=14, pad=10)
            axes[1].set_xlabel(r"Normalized Value ($z$-score)", fontsize=14)
            axes[1].set_ylabel("Frequency (Log Scale)", fontsize=14)
            axes[1].grid(True, which="both", ls="--", alpha=0.3)
            add_stats_box(axes[1], Y_data.flatten(), '#FF8C00')
            add_info_box(axes[1], set_name)

            # Save
            plt.savefig(os.path.join(norm_dir, f"hist_{set_name}_improved.png"), dpi=300)
            plt.close()

        # Generate plots for all splits
        plot_hist_improved(X_train_n, Y_train_n, "Train")
        plot_hist_improved(X_val_n,   Y_val_n,   "Validation")
        plot_hist_improved(X_test_n,  Y_test_n,  "Test")

        # 3. Store Results (JAX Arrays)
        results[stage_tag] = (
            jnp.array(X_train_n), jnp.array(X_val_n), jnp.array(X_test_n),
            jnp.array(Y_train_n), jnp.array(Y_val_n), jnp.array(Y_test_n),
            X_mean, X_std, Y_mean, Y_std
        )
    return results

# ===========================================================
# 2. Save/load checkpoint utilities
# ===========================================================
def save_checkpoint(params, X_mean, X_std, Y_mean, Y_std, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    to_save = {"params": params, "X_mean": X_mean, "X_std": X_std, "Y_mean": Y_mean, "Y_std": Y_std}
    with open(path, "wb") as f:
        f.write(flax.serialization.to_bytes(to_save))

def load_checkpoint(path, init_params):
    with open(path, "rb") as f:
        restored = flax.serialization.from_bytes(init_params, f.read())
    return restored

# ===========================================================
# 3. Plotting utilities (unchanged from stable)
# ===========================================================
def plot_all_losses(train_tot, val_tot, train_d, val_d, train_p, val_p, Y_std, fig_dir, model_type):
    Y_var = float(np.mean(np.array(Y_std)**2))
    #train_tot_ph = np.array(train_tot) * Y_var
    #val_tot_ph = np.array(val_tot) * Y_var
    train_d_ph = np.array(train_d) * Y_var
    val_d_ph = np.array(val_d) * Y_var
    train_p_ph = np.array(train_p) * Y_var
    val_p_ph = np.array(val_p) * Y_var

    def smooth_curve(vals, w=0.9):
        smoothed, last = [], vals[0]
        for v in vals:
            last = last*w + (1-w)*v
            smoothed.append(last)
        return smoothed

    #train_tot_ph = smooth_curve(train_tot_ph)
    #val_tot_ph = smooth_curve(val_tot_ph)
    train_d_ph = smooth_curve(train_d_ph)
    val_d_ph = smooth_curve(val_d_ph)
    train_p_ph = smooth_curve(train_p_ph)
    val_p_ph = smooth_curve(val_p_ph)

    plt.figure(figsize=(10, 6))
    #plt.plot(train_tot_ph, label="Train Total", lw=2)
    #plt.plot(val_tot_ph, label="Val Total", lw=2)
    plt.plot(train_d_ph, label="Train Data", lw=2)
    plt.plot(val_d_ph, label="Val Data", lw=2)
    plt.plot(train_p_ph, label="Train Physics", lw=2)
    plt.plot(val_p_ph, label="Val Physics", lw=2)
    plt.yscale("log")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel(r"MSE Loss [Pa$^2$]", fontsize=14)
    plt.title(f"All Losses ({model_type})", fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, "all_losses_physical.png"), dpi=300)
    plt.close()


def plot_residual_hist(residuals, fig_dir, model_type):
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
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, "residual_histogram.png"))
    plt.close()

def plot_residuals_vs_pred(y_pred, residuals, fig_dir, model_type):
    y_pred_1d, residuals_1d = np.ravel(y_pred), np.ravel(residuals)
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
    plt.axhline(0, color='black', linestyle='--')
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, "residuals_vs_predictions.png"))
    plt.close()

# === stress tensor and summary plotting also copied from stable ===
def plot_stress_tensor_comparison(vec6_to_sym3, y_true_phys, y_pred_phys, sample_indices, fig_dir, model_type):
    """
    Plot 4-panel stress tensor comparison in a 2×2 grid:
      [ True , Predicted ]
      [ Abs Error , Relative Error (%) ]
    Text colour is automatically chosen per-cell for max contrast with background colour.
    """
    os.makedirs(fig_dir, exist_ok=True)
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
        axes[0, 0].set_xticks([])   # remove x-axis ticks
        axes[0, 0].set_yticks([])   # remove y-axis ticks
        fig.colorbar(im0, ax=axes[0, 0], format="%.0e")

        # --- Predicted tensor ---
        im1 = axes[0, 1].imshow(T_pred, cmap="viridis", vmin=common_min, vmax=common_max)
        add_text_with_contrast(axes[0, 1], T_pred, "viridis", fmt="{:.2e}")
        axes[0, 1].set_title(f"Predicted (sample {idx})")
        axes[0, 0].set_xticks([])   # remove x-axis ticks
        axes[0, 0].set_yticks([])   # remove y-axis ticks
        fig.colorbar(im1, ax=axes[0, 1], format="%.0e")

        # --- Absolute Error tensor ---
        im2 = axes[1, 0].imshow(T_err, cmap="RdBu_r", vmin=-err_max, vmax=err_max)
        add_text_with_contrast(axes[1, 0], T_err, "RdBu_r", fmt="{:.2e}")
        axes[1, 0].set_title("Abs Error (True−Pred)")
        axes[0, 0].set_xticks([])   # remove x-axis ticks
        axes[0, 0].set_yticks([])   # remove y-axis ticks
        fig.colorbar(im2, ax=axes[1, 0], format="%.0e")

        # --- Relative Error (%) ---
        im3 = axes[1, 1].imshow(T_relerr, cmap="inferno")
        add_text_with_contrast(axes[1, 1], T_relerr, "inferno", fmt="{:.2f}%")
        axes[1, 1].set_title("Relative Error (%)")
        axes[0, 0].set_xticks([])   # remove x-axis ticks
        axes[0, 0].set_yticks([])   # remove y-axis ticks
        fig.colorbar(im3, ax=axes[1, 1], format="%.2f")

        # Final layout and save
        plt.suptitle(f"{model_type} Stress Tensor Comparison (sample {idx})", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir,
                                 f"stress_tensor_comparison_sample_{idx}.png"),
                    dpi=300)
        plt.close()

def plot_dataset_predictions_summary(vec6_to_sym3, y_true_phys, y_pred_phys, fig_dir, model_type, metrics_table=None):
    """
    Aggregate dataset performance plots:
      - Abs/Rel error histograms side by side
      - True vs Predicted scatter with R²
      - Mean abs/rel error heatmaps per tensor component
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
    # Absolute and relative error arrays
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
# ===========================================================
# 4. Main entry to run loader only
# ===========================================================
@hydra.main(config_path="../config/train", config_name="stable_tensor_config", version_base=None)
def main(cfg: DictConfig):
    stages = [
        "1.0", "1.0_1.2", "1.2_1.4", "1.4_1.6", "1.6_1.8",
        "1.8_2.0", "2.0_2.2", "2.2_2.4", "2.4_2.6", "2.6_2.8"
    ]
    data_root = "datafiles"
    replay_ratio = cfg.data.get("replay_ratio", 0.2)  # <-- default if missing
    print(f"Processing Replay Loader for {cfg.model_type}")
    results = load_and_normalize_stagewise_data_replay(
        model_type=cfg.model_type,
        data_root=data_root,
        stages=stages,
        seed=cfg.seed,
        scaling_mode=cfg.data.scaling_mode,
        replay_ratio=replay_ratio
    )
    print("Processed stages:", list(results.keys()))
    return results

if __name__ == "__main__":
    main()