#!/usr/bin/env python3
"""
Post-Training Analysis Utilities (Flow-Specific)
Responsible for: Loss curves, Global Error Analysis, and Automated Reporting.
Tailored for: Uniaxial, Shear, Planar, and Mixed Flows.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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
# 2. Plotting Functions
# ===========================================================
def plot_all_losses(train_d, val_d, train_p, val_p, Y_std, fig_dir, 
                    model_type, stage_tag, flow_type="Unknown", n_samples=None):
    """
    Plots training and validation loss curves with COMPACT info box.
    Customized for Flow Type labeling.
    """
    
    train_d_ph = np.array(train_d)
    val_d_ph = np.array(val_d)
    train_p_ph = np.array(train_p)
    val_p_ph = np.array(val_p)

    # 2. Smoothing Helper
    def smooth_curve(vals, w=0.95):
        if len(vals) < 2: return vals
        smoothed, last = [], vals[0]
        for v in vals:
            last = last * w + (1 - w) * v
            smoothed.append(last)
        return smoothed

    # Format Names
    clean_flow = flow_type.replace("_", " ").title()

    # 3. Plotting
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    # Plot Lines
    plt.plot(smooth_curve(train_d_ph), label="Train Data", lw=2, color='#1f77b4') 
    plt.plot(smooth_curve(val_d_ph), label="Val Data", lw=2, color='#ff7f0e')     
    plt.plot(smooth_curve(train_p_ph), label="Train Physics", lw=2, color='#2ca02c') 
    plt.plot(smooth_curve(val_p_ph), label="Val Physics", lw=2, color='#d62728')    

    # INFO BOX
    #sz_str = f"{n_samples:,}" if n_samples else "N/A"

    line_handles, line_labels = ax.get_legend_handles_labels()
    
    info_handles = [
        Line2D([0], [0], color='none', label=f'Stage: {stage_tag}')
    ]
    
    final_handles = line_handles + info_handles
    
    plt.legend(handles=final_handles, 
               loc='upper right', fontsize=10, labelspacing=0.4,
               frameon=True, fancybox=True, edgecolor='gray', framealpha=0.95)

    plt.yscale("log")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel(r"MSE Loss [Pa$^2$]", fontsize=14)
    plt.title(f"Training & Validation Loss Curves ({clean_flow})", fontsize=16, pad=12)
    plt.grid(True, which='both', ls='--', lw=0.5, alpha=0.7)

    plt.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, "all_losses_physical.png"), dpi=300)
    plt.close()

# ===========================================================
# plot_global_stress_summary (4-Panel Mean Plot)
# ===========================================================
def plot_global_stress_summary(y_true, y_pred, fig_dir, model_type, flow_type="Unknown"):
    """
    Plots a 4-panel summary of the ENTIRE Test Set (Averaged).
    [Mean True]   [Mean Pred]
    [Mean AbsErr] [Mean RelErr]
    
    Handles flow-specific zero-stress components robustly using Engineering Floor.
    """
    # 1. Convert all vectors to matrices (N, 3, 3)
    T_true_all = vec6_to_sym3(y_true)
    T_pred_all = vec6_to_sym3(y_pred)
    
    # 2. Calculate Global Means
    mean_true = np.mean(T_true_all, axis=0)
    mean_pred = np.mean(T_pred_all, axis=0)
    
    # 3. Calculate Error Means
    # Calculate error for EVERY sample first, then average
    abs_err_all = np.abs(T_true_all - T_pred_all)
    mean_abs_err = np.mean(abs_err_all, axis=0)
    
    # --- ENGINEERING FLOOR LOGIC ---
    # Critical for flow data where off-diagonals or diagonals can be exactly zero.
    FLOOR = 0.005
    
    # Use the floor for the denominator
    denom = np.maximum(np.abs(T_true_all), FLOOR)
    
    # Calculate Relative Error
    rel_err_all = abs_err_all / denom * 100.0
    
    # Take the mean across samples
    mean_rel_err = np.mean(rel_err_all, axis=0)
    # --------------------------------

    # 4. Plotting Setup
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    clean_model = model_type.replace("_", " ").title()
    clean_flow = flow_type.replace("_", " ").title()
    
    # Define configurations for the 4 panels
    # We use common vmin/vmax for the top row to make comparison easy
    top_vmin = min(mean_true.min(), mean_pred.min())
    top_vmax = max(mean_true.max(), mean_pred.max())

    plots_config = [
        # (0,0) Mean True
        {
            "data": mean_true,
            "title": r"Mean True Tensor ($T_{{ij}}$)",
            "cmap": "viridis", "fmt": "{:.2e}",
            "vmin": top_vmin, "vmax": top_vmax
        },
        # (0,1) Mean Pred
        {
            "data": mean_pred,
            "title": r"Mean Predicted Tensor ($T_{{ij}}$)",
            "cmap": "viridis", "fmt": "{:.2e}",
            "vmin": top_vmin, "vmax": top_vmax
        },
        # (1,0) Mean Abs Error
        {
            "data": mean_abs_err,
            "title": "Mean Absolute Error",
            "cmap": "RdBu_r", "fmt": "{:.2e}",
            "vmin": 0, "vmax": np.max(mean_abs_err)
        },
        # (1,1) Mean Rel Error
        {
            "data": mean_rel_err,
            "title": f"Mean Relative Error %",
            "cmap": "magma", "fmt": "{:.1f}%",
            "vmin": 0, "vmax": np.max(mean_rel_err)
        }
    ]

    # Loop to create subplots
    for ax, config in zip(axes.flat, plots_config):
        data = config["data"]
        im = ax.imshow(data, cmap=config["cmap"], vmin=config["vmin"], vmax=config["vmax"])
        ax.set_title(config["title"], fontsize=12, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])

        # Annotations
        for i in range(3):
            for j in range(3):
                val = data[i, j]
                # Contrast logic
                text_color = "white"
                # Brighter backgrounds need black text
                if config["cmap"] == "viridis" and val > top_vmax * 0.7: text_color = "black"
                if config["cmap"] == "magma" and val > config["vmax"] * 0.7: text_color = "black"
                
                ax.text(j, i, config["fmt"].format(val), 
                        ha="center", va="center", 
                        color=text_color, fontweight="bold", fontsize=9)

        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im, cax=cax)

    # UPDATED TITLE for Flow
    plt.suptitle(f"{clean_model} ({clean_flow}): Global Test Set Summary", fontsize=15, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    save_path = os.path.join(fig_dir, "global_stress_tensor_summary.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

# ===========================================================
# plot_dataset_predictions_summary (Histograms + Logs + Scatter)
# ===========================================================
def plot_dataset_predictions_summary(y_true_phys, y_pred_phys, fig_dir, shared_log_dir, 
                                     model_type, metrics_table=None, seed=None, 
                                     log_filename=None, n_samples=None, 
                                     stage_tag="Unknown", flow_type="Unknown", elapsed_time=None): # <--- Added Arg
    """
    1. Saves Plots (Histograms, Scatter) to the specific run folder (fig_dir).
    2. Appends Text Metrics to a central file specific to the flow type.
    """
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(shared_log_dir, exist_ok=True)
    labels = ["σ_xx", "σ_yy", "σ_zz", "σ_xy", "σ_xz", "σ_yz"]
    
    # --- ENGINEERING FLOOR ---
    FLOOR = 0.005
    
    clean_model = model_type.replace("_", " ").title()
    clean_flow = flow_type.replace("_", " ").title()

    # --- 1. Append Metrics to Text File ---
    # NOTE: Filename now includes FLOW TYPE to separate logs
    if log_filename is None: fname = f"{model_type}_{flow_type}_metrics.txt"
    else: fname = log_filename
    metrics_path = os.path.join(shared_log_dir, fname)

    # =========================================================
    # CALCULATE METRICS (Global)
    # =========================================================
    
    # A. Floored MRE (Replaces the filtered threshold logic)
    denom_global = np.maximum(np.abs(y_true_phys), FLOOR)
    raw_rel = np.abs(y_true_phys - y_pred_phys) / denom_global
    filtered_mre_val = np.mean(raw_rel) * 100.0

    # B. Global Tensor Norm Error (Preserved as requested)
    # Error = ||T_pred - T_true|| / ||T_true||
    norm_true = np.linalg.norm(y_true_phys, axis=1)
    norm_err  = np.linalg.norm(y_true_phys - y_pred_phys, axis=1)
    # Add epsilon to norm_true to prevent division by zero in rare resting cases
    global_norm_err_val = np.mean((norm_err / (norm_true + 1e-6)) * 100.0)
    
    # Format time string
    time_str = f"{elapsed_time:.2f}s" if elapsed_time is not None else "N/A"

    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f" RUN SUMMARY | Model: {model_type} | Flow: {flow_type} | Time: {time_str} | Seed: {seed} | Size: {n_samples} | Date: {np.datetime64('now')}\n")
        f.write("="*80 + "\n\n")

        f.write(f"--- Component-wise Breakdown (Seed {seed}) ---\n")
        header = "{:<10} {:>12} {:>12} {:>8} {:>15} {:>18} {:>18}\n".format(
            "Component", "MSE", "RMSE", "R²", "MeanAbsErr", "MeanRelErr(%)", "N_Samples"
        )
        f.write(header)
        f.write("-" * len(header) + "\n")

        for k in range(6):
            # Standard Metrics
            mse = mean_squared_error(y_true_phys[:, k], y_pred_phys[:, k])
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true_phys[:, k], y_pred_phys[:, k])
            abs_err_comp = np.abs(y_true_phys[:, k] - y_pred_phys[:, k])
            mean_abs_err_val = np.mean(abs_err_comp)
            
            # --- FIXED: Floored Relative Error ---
            denom_comp = np.maximum(np.abs(y_true_phys[:, k]), FLOOR)
            rel_err_comp = abs_err_comp / denom_comp * 100.0
            mean_rel_err_val = np.mean(rel_err_comp)
            # -------------------------------------
            
            n_samples_test = y_true_phys.shape[0]

            f.write("{:<10} {:>12.4e} {:>12.4e} {:>8.4f} {:>15.4e} {:>18.2f} {:>18}\n".format(
                labels[k], mse, rmse, r2, mean_abs_err_val, mean_rel_err_val, n_samples_test
            ))
        f.write("\n")

        # =========================================================
        # WRITE NEW METRICS
        # =========================================================
        f.write(f"--- Global Performance Metrics ---\n")
        f.write(f"✅ Global Floored MRE (Floor={FLOOR}):   {filtered_mre_val:.4f}%\n")
        f.write(f"✅ Global Tensor Norm Error:        {global_norm_err_val:.4f}%\n\n")

        if metrics_table is not None:
            f.write(f"--- Overall Metrics (Seed {seed}) ---\n")
            grid_table = tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="grid")
            f.write(grid_table)
            f.write("\n\n")

    # --- 2. Histograms (Side-by-side) ---
    abs_err = np.abs(y_true_phys - y_pred_phys)
    
    # Histogram uses same floor logic for consistency
    denom_hist = np.maximum(np.abs(y_true_phys), FLOOR)
    rel_err_floored = abs_err / denom_hist * 100.0

    sz_str = f"{n_samples:,}" if n_samples else "N/A"

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Absolute
    abs_limit = np.percentile(abs_err, 99)
    ax[0].hist(abs_err.flatten(), bins=50, range=(0, abs_limit), color='skyblue', edgecolor='black')
    ax[0].set_xlabel("Absolute Error Value")
    ax[0].set_ylabel("Frequency")
    ax[0].grid(True, ls="--", alpha=0.5)
    
    stats_text_abs = (
        f"Test Set Results\n"
        f"{'Mean:':<7} {np.mean(abs_err):.2e}\n"
        f"{'Median:':<7} {np.median(abs_err):.2e}"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    
    ax[0].text(0.95, 0.95, stats_text_abs, transform=ax[0].transAxes, fontsize=9,
               verticalalignment='top', horizontalalignment='right', multialignment='left', 
               bbox=props, family='monospace') 

    # Relative (Floored)
    rel_limit = np.percentile(rel_err_floored, 99)
    ax[1].hist(rel_err_floored.flatten(), bins=50, range=(0, rel_limit), color='orange', edgecolor='black', alpha=0.9)
    ax[1].set_xlabel(f"Relative Error %")
    ax[1].set_ylabel("Frequency")
    ax[1].grid(True, ls="--", alpha=0.5)
    
    stats_text_rel = (
        f"Test Set Results\n"
        f"{'Mean:':<7} {np.mean(rel_err_floored):.2f}%\n"
        f"{'Median:':<7} {np.median(rel_err_floored):.2f}%"
    )
    ax[1].text(0.95, 0.95, stats_text_rel, transform=ax[1].transAxes, fontsize=9,
               verticalalignment='top', horizontalalignment='right', multialignment='left', 
               bbox=props, family='monospace')

    # UPDATED TITLE for Flow
    plt.suptitle(f"{clean_model} ({clean_flow}): Component-wise Error Distribution ($T_{{ij}}$)", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(os.path.join(fig_dir, "abs_rel_error_hist.png"), dpi=300)
    plt.close()

    # --- 3. Scatter Plot ---
    plt.figure(figsize=(6,6))
    plt.scatter(y_true_phys.flatten(), y_pred_phys.flatten(), s=5, alpha=0.5)
    plt.plot([y_true_phys.min(), y_true_phys.max()], [y_true_phys.min(), y_true_phys.max()], 'k--')
    
    # INFO BOX (Top Left)
    stats_text_scatter = (
        f"Test Set Results\n"
        f"Stage: {stage_tag}"
    )
    plt.gca().text(0.05, 0.95, stats_text_scatter, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', horizontalalignment='left', bbox=props, family='monospace')

    plt.xlabel("True Stress Components ($T_{{ij}}$)")
    plt.ylabel("Predicted Stress Components ($T_{{ij}}$)")
    plt.title(f"{clean_model} ({clean_flow}) - True vs Predicted")
    plt.grid(True, ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "true_vs_pred_scatter.png"), dpi=300)
    plt.close()

# ===========================================================
# 4. Main Orchestrator
# ===========================================================
def run_post_training_analysis(
    y_true_n, y_pred_n, 
    Y_mean, Y_std, 
    stage_tag, model_type, base_dir, 
    flow_type="Unknown", n_samples=None
):
    """
    Standard entry point for validation (Flow-Specific).
    Optimized: Removes redundant plots, focuses on Global Summaries.
    """
    # 1. Denormalize
    Y_mean = np.array(Y_mean)
    Y_std = np.array(Y_std)
    Y_std_safe = np.where(Y_std == 0, 1.0, Y_std)
    y_true = (y_true_n * Y_std_safe) + Y_mean
    y_pred = (y_pred_n * Y_std_safe) + Y_mean
    
    # 2. Set Output Directory (Directly use the passed folder)
    fig_dir = base_dir 
    os.makedirs(fig_dir, exist_ok=True)
    
    # 3. Global Stress Summary (The 4-panel Heatmap)
    try:
        plot_global_stress_summary(y_true, y_pred, fig_dir, model_type, flow_type=flow_type)
    except Exception as e:
        print(f"⚠️ Global Summary plot failed: {e}")

    # 4. Dataset Predictions Summary (Histograms + Scatter + Logs)
    plot_dataset_predictions_summary(y_true, y_pred, fig_dir, model_type, 
                                     n_samples=n_samples, stage_tag=stage_tag, 
                                     flow_type=flow_type)

if __name__ == "__main__":
    print("Post-training analysis module loaded (Flow-Specific).")