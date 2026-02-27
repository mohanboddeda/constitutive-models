#!/usr/bin/env python3
"""
Post-Training Analysis Utilities
Responsible for: Loss curves, Residual Analysis, Stress Tensor Visualization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tabulate import tabulate
from matplotlib.lines import Line2D

# ===========================================================
# 1. Helper Utilities
# ===========================================================
def vec6_to_sym3(v):
    """
    Converts (N, 6) vector [xx, yy, zz, xy, xz, yz] -> (N, 3, 3) symmetric matrix.
    """
    v = np.array(v)
    N = v.shape[0]
    T = np.zeros((N, 3, 3))
    
    # Diagonals: xx=0, yy=1, zz=2
    T[:, 0, 0] = v[:, 0]
    T[:, 1, 1] = v[:, 1]
    T[:, 2, 2] = v[:, 2]
    
    # Off-diagonals: xy=3, xz=4, yz=5
    T[:, 0, 1] = v[:, 3]; T[:, 1, 0] = v[:, 3]
    T[:, 0, 2] = v[:, 4]; T[:, 2, 0] = v[:, 4]
    T[:, 1, 2] = v[:, 5]; T[:, 2, 1] = v[:, 5]
    return T

# ===========================================================
# 2. Plotting Functions (Loss, Residuals, Tensors)
# ===========================================================
# ===========================================================
# plot_all_losses
# ===========================================================
# ===========================================================
# plot_all_losses (UPDATED)
# ===========================================================
def plot_all_losses(train_d, val_d, train_p, val_p, Y_std, fig_dir, model_type, stage_tag, n_samples=None):
    """
    Plots training and validation loss curves with COMPACT info box.
    Format:
    [Line] Train Data
    [Line] Val Data
    [Line] Train Physics
    [Line] Val Physics
    Model: Maxwell B
    Stage: 1.0_1.2
    Sample Size: 10000
    """
    # 1. Un-normalize to physical units
    Y_var = float(np.mean(np.array(Y_std)**2))
    train_d_ph = np.array(train_d) * Y_var
    val_d_ph = np.array(val_d) * Y_var
    train_p_ph = np.array(train_p) * Y_var
    val_p_ph = np.array(val_p) * Y_var

    # 2. Smoothing Helper
    def smooth_curve(vals, w=0.95):
        if len(vals) < 2: return vals
        smoothed, last = [], vals[0]
        for v in vals:
            last = last * w + (1 - w) * v
            smoothed.append(last)
        return smoothed

    # 3. Plotting
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    # Plot Lines (These will be the first 4 items in the legend)
    plt.plot(smooth_curve(train_d_ph), label="Train Data", lw=2, color='#1f77b4') 
    plt.plot(smooth_curve(val_d_ph), label="Val Data", lw=2, color='#ff7f0e')     
    plt.plot(smooth_curve(train_p_ph), label="Train Physics", lw=2, color='#2ca02c') 
    plt.plot(smooth_curve(val_p_ph), label="Val Physics", lw=2, color='#d62728')    

    # ---------------------------------------------------------
    # CREATE THE COMPACT INFO BOX
    # ---------------------------------------------------------
    clean_model = model_type.replace("_", " ").title()
    
    # Format Sample Size string (e.g., "10,000" or "N/A")
    if n_samples:
        sz_str = f"{n_samples:,}" # Adds comma: 10,000
    else:
        sz_str = "N/A"

    # 1. Get the actual plot handles (The 4 colored lines)
    line_handles, line_labels = ax.get_legend_handles_labels()
    
    # 2. Create "Invisible" handles for the text info (Bottom section)
    # Note: 'label' contains the text we want to show
    info_handles = [
        Line2D([0], [0], color='none', label=f'Model: {clean_model}'),
        Line2D([0], [0], color='none', label=f'Stage: {stage_tag}'),
        Line2D([0], [0], color='none', label=f'Sample Size: {sz_str}')
    ]
    
    # 3. Combine: Lines FIRST, then Info
    final_handles = line_handles + info_handles
    
    # 4. Create the single legend box
    # labelspacing=0.4 tightens the vertical gap
    plt.legend(handles=final_handles, 
               loc='upper right', 
               fontsize=10,        # Slightly smaller for compactness
               labelspacing=0.4,   # Squeezes the rows together
               frameon=True, 
               fancybox=True, 
               edgecolor='gray',
               framealpha=0.95)

    # 4. Formatting
    plt.yscale("log")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel(r"MSE Loss [Pa$^2$]", fontsize=14)
    plt.title("Training & Validation Loss Curves", fontsize=16, pad=12)
    plt.grid(True, which='both', ls='--', lw=0.5, alpha=0.7)

    plt.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, "all_losses_physical.png"), dpi=300)
    plt.close()

# ===========================================================
# plot_residual_hist
# ===========================================================
def plot_residual_hist(residuals, fig_dir, model_type, stage_tag):
    """
    Plots a histogram of Data Residuals with a unified Info Box in the legend.
    """
    residuals_1d = np.ravel(residuals)
    
    plt.figure(figsize=(9, 6))
    ax = plt.gca()

    # 1. Histogram
    # We capture the plot output to ensure the label is registered
    plt.hist(residuals_1d, bins=50, density=True, 
             color='skyblue', edgecolor='black', alpha=1.0,
             label='Histogram')

    # 2. KDE Line
    try:
        kde = gaussian_kde(residuals_1d)
        x_range = np.linspace(residuals_1d.min(), residuals_1d.max(), 1000)
        plt.plot(x_range, kde(x_range), color='orange', lw=2.5, label='KDE')
    except Exception as e:
        print(f"Warning: KDE plot failed: {e}")

    # 3. Zero Residual Line
    plt.axvline(0, color='red', linestyle='--', lw=2.0, label='Zero Residual')

    # 4. Create Unified Info Box (Legend)
    
    # Clean text
    clean_model = model_type.replace("_", " ").title()
    
    # 1. Get existing handles and labels from the plot
    handles, labels = ax.get_legend_handles_labels()
    
    # 2. Create "Fake" handles AND labels for the info text
    info_handles = [
        Line2D([0], [0], color='none'), 
        Line2D([0], [0], color='none'), 
        Line2D([0], [0], color='none')
    ]
    info_labels = [
        f'Model: {clean_model}', 
        f'Stage: {stage_tag}', 
        ''  # Spacer
    ]
    
    # 3. Combine them
    final_handles = info_handles + handles
    final_labels = info_labels + labels

    # 4. Pass BOTH to legend
    plt.legend(handles=final_handles, labels=final_labels, 
               loc='upper right', frameon=True, fancybox=True, 
               framealpha=0.95, edgecolor='gray')

    # 5. Formatting
    plt.title(f"Data Residuals Histogram", fontsize=15, pad=10)
    plt.xlabel("Residual Data Error (True - Predicted)", fontsize=13)
    plt.ylabel("Density", fontsize=13)
    plt.grid(True, which='both', ls='-', alpha=0.6)
    
    plt.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, "residual_histogram.png"), dpi=300)
    plt.close()

# ===========================================================
# plot_stress_tensor_comparison
# ===========================================================
def plot_stress_tensor_comparison(T_true, T_pred, sample_idx):
    """
    Plots a single 2x2 figure for a specific sample.
    Structure:
    [True]      [Predicted]
    [Abs Error] [Rel Error]
    """
    
    # 1. Calculate Metrics
    diff = T_true - T_pred
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_error = np.abs(diff / T_true) * 100
    rel_error = np.nan_to_num(rel_error)

    # 2. Setup Figure (One single plot containing 4 subplots)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Configuration for the 4 subplots
    plots_config = [
        # (0,0) True
        {
            "data": T_true,
            "title": f"True Component Values ($T_{{ij}}$)",
            "cmap": "viridis",
            "fmt": "{:.2e}",
            "vmin": min(T_true.min(), T_pred.min()),
            "vmax": max(T_true.max(), T_pred.max())
        },
        # (0,1) Predicted
        {
            "data": T_pred,
            "title": f"Predicted Component Values ($T_{{ij}}$)",
            "cmap": "viridis",
            "fmt": "{:.2e}",
            "vmin": min(T_true.min(), T_pred.min()),
            "vmax": max(T_true.max(), T_pred.max())
        },
        # (1,0) Abs Error
        {
            "data": diff,
            "title": "Absolute Error",
            "cmap": "RdBu_r",
            "fmt": "{:.2e}",
            "vmin": -np.max(np.abs(diff)), # Centered at 0
            "vmax": np.max(np.abs(diff))
        },
        # (1,1) Rel Error
        {
            "data": rel_error,
            "title": "Relative Error (%)",
            "cmap": "magma",
            "fmt": "{:.2f}%",
            "vmin": 0,
            "vmax": np.max(rel_error)
        }
    ]

    # 3. Plotting Loop
    for ax, config in zip(axes.flat, plots_config):
        data = config["data"]
        
        # Plot Heatmap
        im = ax.imshow(data, cmap=config["cmap"], vmin=config["vmin"], vmax=config["vmax"])
        ax.set_title(config["title"], fontsize=12, pad=10)
        
        # --- Remove Axes (Ticks and Labels) ---
        ax.set_xticks([])
        ax.set_yticks([])

        # --- Annotations ---
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                # Text color logic for readability
                text_color = "white"
                if config["cmap"] == "RdBu_r" and abs(val) < config["vmax"] * 0.3: text_color = "black"
                elif config["cmap"] == "magma" and val > config["vmax"] * 0.7: text_color = "black"
                
                ax.text(j, i, config["fmt"].format(val), 
                        ha="center", va="center", 
                        color=text_color, fontweight="bold", fontsize=9)

        # --- Adjust Sidebar Height (Colorbar) ---
        # This ensures the colorbar is exactly the same height as the matrix
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im, cax=cax)

    # Global Title
    fig.suptitle(f"Maxwell_B Stress Tensor Analysis - Sample {sample_idx}", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92) # Space for title
    
    return fig

# ===========================================================
# plot_dataset_predictions_summary (Universal Logging)
# ===========================================================
def plot_dataset_predictions_summary(y_true_phys, y_pred_phys, fig_dir, shared_log_dir, 
                                     model_type, metrics_table=None, seed=None, 
                                     log_filename=None, n_samples=None):
    """
    1. Saves Plots (Histograms, Heatmaps) to the specific run folder (fig_dir).
    2. Appends Text Metrics to a central file in shared_log_dir.
    
    Parameters:
    - log_filename: Name of the text file to write to. 
                    e.g. "maxwell_B_single_stage_metrics.txt"
    """
    # 1. Create Directories
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(shared_log_dir, exist_ok=True)
    
    labels = ["σ_xx", "σ_yy", "σ_zz", "σ_xy", "σ_xz", "σ_yz"]
    eps = 1e-12

    # 2. Determine Log File Name
    # If None, default to a generic name
    if log_filename is None:
        fname = f"{model_type}_metrics.txt"
    else:
        fname = log_filename
        
    metrics_path = os.path.join(shared_log_dir, fname)

    # 3. Append Metrics to Text File
    with open(metrics_path, "a", encoding="utf-8") as f:
        # Run Header
        f.write("\n" + "="*80 + "\n")
        f.write(f" RUN SUMMARY | Model: {model_type} | Seed: {seed} | Sample Size: {n_samples} | Date: {np.datetime64('now')}\n")
        f.write("="*80 + "\n\n")

        # Part A: Component-wise Breakdown
        f.write(f"--- Component-wise Breakdown of total test set (Seed {seed}) ---\n")
        header = "{:<10} {:>12} {:>12} {:>8} {:>15} {:>18} {:>18}\n".format(
            "Component", "MSE", "RMSE", "R²", "MeanAbsErr", "MeanRelErr(%)", "N_Samples"
        )
        f.write(header)
        f.write("-" * len(header) + "\n")

        for k in range(6):
            mse = mean_squared_error(y_true_phys[:, k], y_pred_phys[:, k])
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true_phys[:, k], y_pred_phys[:, k])
            abs_err_comp = np.abs(y_true_phys[:, k] - y_pred_phys[:, k])
            mean_abs_err_val = np.mean(abs_err_comp)
            rel_err_comp = abs_err_comp / (np.abs(y_true_phys[:, k]) + eps) * 100.0
            mean_rel_err_val = np.mean(rel_err_comp)
            n_samples = y_true_phys.shape[0]

            f.write("{:<10} {:>12.4e} {:>12.4e} {:>8.4f} {:>15.4e} {:>18.2f} {:>18}\n".format(
                labels[k], mse, rmse, r2, mean_abs_err_val, mean_rel_err_val, n_samples
            ))
        f.write("\n")

        # Part B: Overall Metrics (Grid Table)
        if metrics_table is not None:
            f.write(f"--- Overall Metrics (Seed {seed}) ---\n")
            grid_table = tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="grid")
            f.write(grid_table)
            f.write("\n\n")

    # 2. Histograms (Side-by-side)
    # Calculate Errors
    abs_err = np.abs(y_true_phys - y_pred_phys)
    rel_err = abs_err / (np.abs(y_true_phys) + eps) * 100

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Format sample count string
    sz_str = f"{n_samples:,}" if n_samples else "N/A"
    
    # --- Plot 1: Absolute Error ---
    abs_limit = np.percentile(abs_err, 99)
    ax[0].hist(abs_err.flatten(), bins=50, range=(0, abs_limit), color='skyblue', edgecolor='black')
    
    # No individual title needed
    ax[0].set_xlabel("Absolute Error Value")
    ax[0].set_ylabel("Frequency")
    ax[0].grid(True, ls="--", alpha=0.5)
    
    # INFO BOX (Aligned + Monospace)
    stats_text_abs = (
        f"Test Set Results\n"
        f"{'-'*20}\n"
        f"{'Mean:':<10} {np.mean(abs_err):>10.2e}\n"
        f"{'Median:':<10} {np.median(abs_err):>10.2e}\n"
        f"{'N_samples:':<10} {sz_str:>10}"
    )
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax[0].text(0.95, 0.95, stats_text_abs, transform=ax[0].transAxes, fontsize=9,
               verticalalignment='top', horizontalalignment='right', bbox=props, 
               family='monospace') 

    # --- Plot 2: Relative Error Histogram ---
    rel_limit = np.percentile(rel_err, 99)
    ax[1].hist(rel_err.flatten(), bins=50, range=(0, rel_limit), 
               color='orange', edgecolor='black', alpha=0.9)
    
    # No individual title needed
    ax[1].set_xlabel("Relative Error (%)")
    ax[1].set_ylabel("Frequency")
    ax[1].grid(True, ls="--", alpha=0.5)
    
    # INFO BOX
    stats_text_rel = (
        f"Test Set Results\n"
        f"{'-'*20}\n"
        f"{'Mean:':<10} {np.mean(rel_err):>9.2f}%\n"
        f"{'Median:':<10} {np.median(rel_err):>9.2f}%\n"
        f"{'N_samples:':<10} {sz_str:>10}"
    )
    
    ax[1].text(0.95, 0.95, stats_text_rel, transform=ax[1].transAxes, fontsize=9,
               verticalalignment='top', horizontalalignment='right', bbox=props, 
               family='monospace')

    # --- UNIFIED GLOBAL TITLE ---
    plt.suptitle("Maxwell B: Component-wise Error Distribution ($T_{{ij}}$)", fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88) # Make space for the big title
    
    plt.savefig(os.path.join(fig_dir, "abs_rel_error_hist.png"), dpi=300)
    plt.close()

    # 3. Scatter Plot
    plt.figure(figsize=(6,6))
    plt.scatter(y_true_phys.flatten(), y_pred_phys.flatten(), s=5, alpha=0.5)
    # Calculate R2 for the plot title
    #corr_coef = np.corrcoef(y_true_phys.flatten(), y_pred_phys.flatten())[0,1]
    plt.plot([y_true_phys.min(), y_true_phys.max()], [y_true_phys.min(), y_true_phys.max()], 'k--')
    plt.xlabel("True Stress Components ($T_{{ij}}$)")
    plt.ylabel("Predicted Stress Components ($T_{{ij}}$)")
    plt.title("Maxwell_B - True vs Predicted ($T_{{ij}}$)")
    plt.grid(True, ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "true_vs_pred_scatter.png"), dpi=300)
    plt.close()

    # 4. Heatmaps with Contrast Text
    # Calculate mean errors per tensor component
    true_tensors = vec6_to_sym3(y_true_phys)
    pred_tensors = vec6_to_sym3(y_pred_phys)
    
    mean_abs_err_mat = np.mean(np.abs(true_tensors - pred_tensors), axis=0)
    mean_rel_err_mat = np.mean(np.abs(true_tensors - pred_tensors) / (np.abs(true_tensors) + eps) * 100, axis=0)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # Helper function for text contrast
    def add_text_with_contrast(axh, matrix, cmap_name, fmt="{:.2e}"):
        norm = mpl.colors.Normalize(vmin=np.min(matrix), vmax=np.max(matrix))
        cmap = plt.get_cmap(cmap_name)
        for (i, j), val in np.ndenumerate(matrix):
            r, g, b, _ = cmap(norm(val))
            brightness = 0.299*r + 0.587*g + 0.114*b
            text_color = "black" if brightness > 0.6 else "white"
            axh.text(j, i, fmt.format(val), ha='center', va='center',
                     color=text_color, fontsize=9, fontweight='bold')

    # Mean Abs Error Heatmap
    im0 = ax[0].imshow(mean_abs_err_mat, cmap="RdBu_r")
    ax[0].set_title("Mean Absolute Error")
    fig.colorbar(im0, ax=ax[0])
    add_text_with_contrast(ax[0], mean_abs_err_mat, "RdBu_r", fmt="{:.2e}")
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    # Mean Rel Error Heatmap
    im1 = ax[1].imshow(mean_rel_err_mat, cmap="inferno")
    ax[1].set_title("Mean Relative Error (%)")
    fig.colorbar(im1, ax=ax[1])
    add_text_with_contrast(ax[1], mean_rel_err_mat, "inferno", fmt="{:.1f}%")
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    plt.suptitle("Maxwell_B Stress Tensor Errors", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "mean_error_heatmaps.png"), dpi=300)
    plt.close()

# ===========================================================
# 5. Main Orchestrator
# ===========================================================
def run_post_training_analysis(
    y_true_n, y_pred_n, 
    Y_mean, Y_std, 
    stage_tag, model_type, base_dir
):
    """
    Standard entry point for validation.
    SILENT MODE: Does not print start/end messages to console.
    SINGLE FOLDER MODE: Saves images directly to 'base_dir' (which is the stage folder).
    """
    # 1. Denormalize
    Y_mean = np.array(Y_mean)
    Y_std = np.array(Y_std)
    Y_std_safe = np.where(Y_std == 0, 1.0, Y_std)
    y_true = (y_true_n * Y_std_safe) + Y_mean
    y_pred = (y_pred_n * Y_std_safe) + Y_mean
    residuals = y_true - y_pred
    
    # 2. Set Output Directory (Directly use the passed folder)
    fig_dir = base_dir 
    os.makedirs(fig_dir, exist_ok=True)
    
    # 3. Run Global Plots
    plot_residual_hist(residuals, fig_dir, model_type, stage_tag)
    plot_dataset_predictions_summary(y_true, y_pred, fig_dir, model_type)
    
    # 4. Sample Analysis (Best, Median, Worst)
    # Calculate MSE per sample to find best/median/worst
    mse_per_sample = np.mean(residuals**2, axis=1)
    
    idx_best = np.argmin(mse_per_sample)
    idx_worst = np.argmax(mse_per_sample)
    # Median based on sorting MSE
    idx_median = np.argsort(mse_per_sample)[len(mse_per_sample)//2]
    
    indices_to_plot = [idx_best, idx_median, idx_worst]
    
    # Loop through the 3 selected samples
    for idx in indices_to_plot:
        # Extract single vector and convert to 3x3 matrix
        # Note: vec6_to_sym3 expects (N, 6), so we slice [idx:idx+1]
        t_true = vec6_to_sym3(y_true[idx:idx+1])[0] 
        t_pred = vec6_to_sym3(y_pred[idx:idx+1])[0]
        
        # Plot
        fig = plot_stress_tensor_comparison(t_true, t_pred, sample_idx=idx)
        
        # Save individually
        save_path = os.path.join(fig_dir, f"stress_tensor_sample_{idx}.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

if __name__ == "__main__":
    # Example usage just to test the file runs
    print("Post-training analysis module loaded.")