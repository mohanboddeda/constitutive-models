#!/usr/bin/env python3
"""
utils/pretrain_uniform_net2net.py
---------------------------------
Specialized Data Loader for Net2Net + Uniform Training.

DIFFERENCES FROM ORIGINAL (pretrain_uniform.py):
1. ANCHOR NORMALIZATION: 
   - Calculates Mean/Std ONLY from the first stage (Stage 1.0_1.8).
   - All subsequent stages (and Replay Data) are forced to use these fixed anchors.
   - This ensures T=50 always means the same input value to the network, preventing 
     "concept drift" when the distribution shifts towards high-stress outliers.

2. FLEXIBLE CHECKPOINTING: 
   - `load_checkpoint` accepts `init_params=None` to return a raw dictionary.
   - This prevents shape mismatch errors when loading 'Small' weights (64 neurons) 
     into a script initialized for 'Big' weights (256 neurons).

3. REPLAY MEMORY: 
   - Retains 20% of data from previous stages to prevent catastrophic forgetting.
   - Replay data is normalized using the ANCHOR stats, not its own stats.
"""

import os
import sys
import time
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import flax
from flax import serialization

# --- FIX: Add project root to path so 'utils' module can be found ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# --------------------------------------------------------------------

from utils.invariants import compute_invariants_vectorized

# ===========================================================
# 1. Helper: Invariant Space Visualization (Replay Mode)
# ===========================================================
def plot_invariants_replay_mode(X_curr, X_replay, stage_tag, save_dir, model_name):
    """
    Plots the invariant space (II vs III) to visualize the Replay Buffer.
    
    Visual Elements:
    - Background: Lumley Triangle (Admissible=Blue, Inadmissible=Pink).
    - Red Points: Samples from the CURRENT stage.
    - Blue Points: Samples from PAST stages (Replay Memory).
    """
    # Create figure
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    
    # --- A. Prepare Data (Convert Velocity Gradient L -> Rate of Strain D) ---
    def get_invariants(X_data):
        # Ensure shape is (N, 3, 3)
        if X_data.ndim == 2:
            X_data = X_data.reshape(-1, 3, 3)
        
        # Calculate Symmetric part D = 0.5 * (L + L.T)
        D = 0.5 * (X_data + np.swapaxes(X_data, 1, 2))
        
        # Compute invariants (Vectorized)
        _, II, III = compute_invariants_vectorized(D)
        return -II, III  # Plotting -II on X-axis

    # Extract invariants
    II_curr, III_curr = get_invariants(X_curr)
    II_repl, III_repl = get_invariants(X_replay)

    # --- B. Background (Lumley Triangle) ---
    II_vals = np.linspace(0, 1.6, 400)
    III_vals = np.linspace(-0.7, 0.7, 400) 
    II_grid, III_grid = np.meshgrid(II_vals, III_vals)
    
    # Discriminant condition
    discriminant = ((-III_grid / 2)**2 + (-II_grid / 3)**3)

    # Contour Plot: Pink (#ffa8a8) = Inadmissible, Blue (#a8c6ff) = Admissible
    plt.contourf(II_grid, III_grid, discriminant, levels=[-1e10, 0, 1e10],
                 colors=['#ffa8a8', '#a8c6ff'], alpha=0.8, zorder=0)

    # Draw Boundary Lines
    x_b = np.linspace(0, 1.6, 1000)
    y_b = 2 * np.sqrt((x_b/3)**3)
    plt.plot(x_b, y_b, 'k-', lw=1.5, zorder=1)
    plt.plot(x_b, -y_b, 'k-', lw=1.5, zorder=1)
    plt.axhline(0, color='k', lw=1.5, zorder=1)

    # --- C. Scatter Points ---
    # 1. Current Stage (Red)
    plt.scatter(II_curr, III_curr, c='red', s=30, alpha=0.6, 
                edgecolors='k', linewidth=0.5, zorder=2)
    
    # 2. Replay Buffer (Blue)
    plt.scatter(II_repl, III_repl, c='blue', s=30, alpha=0.6, 
                edgecolors='k', linewidth=0.5, zorder=3)

    # --- D. Formatting & Legend ---
    clean_model = model_name.replace("_", " ").title()
    
    plt.title(f"Invariant Space: Replay Mode (Stage {stage_tag})", fontsize=16, pad=10)
    
    plt.xlabel(r"Second Invariant ($-II_{\mathbf{D}}$)", fontsize=14)
    plt.ylabel(r"Third Invariant ($III_{\mathbf{D}}$)", fontsize=14)
    plt.xlim(0, 1.5)
    plt.ylim(-0.6, 0.6)

    # Calculate counts for the legend
    n_curr = len(X_curr)
    n_replay = len(X_replay)

    # Custom Legend with Counts
    legend_elements = [
        Patch(facecolor='#ffa8a8', edgecolor='none', label='Admissible Region'),
        
        # Current Stage with Count
        Line2D([0], [0], marker='o', color='w', label=f'Current Stage (n={n_curr})',
               markerfacecolor='red', markersize=8, markeredgecolor='k'),
        
        # Replay Buffer with Count
        Line2D([0], [0], marker='o', color='w', label=f'Replay Buffer (n={n_replay})',
               markerfacecolor='blue', markersize=8, markeredgecolor='k'),
        
        # Metadata
        Line2D([0], [0], color='none', label=f'Model: {clean_model} (Uniform)'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.95, 
              edgecolor='#cccccc', fontsize=10)
    
    plt.grid(True, linestyle='--', alpha=0.4, zorder=0)
    plt.tight_layout()

    # Save
    out_path = os.path.join(save_dir, f"Replay_Invariants_{stage_tag}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"   📊 Replay Invariant Plot saved: {out_path}")

# ===========================================================
# 2. Flexible Checkpoint Loader (CRITICAL NET2NET FIX)
# ===========================================================
def load_checkpoint(path, init_params=None):
    """
    Loads model parameters from a msgpack file.
    
    CRITICAL MODIFICATION FOR NET2NET:
    - If `init_params` is provided (Standard Mode): It enforces strict shape matching.
    - If `init_params` is None (Net2Net Mode): It returns the raw dictionary.
    
    This allows us to load 'Small' weights (e.g., 64x64) into a script that expects
    'Large' weights (e.g., 256x256) without crashing, so we can then apply padding.
    """
    with open(path, "rb") as f:
        data = f.read()
    
    # 1. Restore Raw Data (MsgPack)
    # We use msgpack_restore to get the raw dict structure first
    restored = serialization.msgpack_restore(data)
    
    # 2. If strict structure provided, use it (Standard behavior)
    if init_params is not None:
        return serialization.from_state_dict(init_params, restored)
    
    # 3. If NO structure (Net2Net mode), return raw dict
    return restored

def save_checkpoint(params, X_mean, X_std, Y_mean, Y_std, path):
    """Saves model parameters and normalization stats via Flax."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    to_save = {
        "params": params, 
        "X_mean": X_mean, "X_std": X_std, 
        "Y_mean": Y_mean, "Y_std": Y_std
    }
    with open(path, "wb") as f:
        f.write(serialization.msgpack_serialize(to_save))

# ===========================================================
# 3. Stage-wise Replay Loader (Main Logic with Anchor Stats)
# ===========================================================
def load_and_normalize_stagewise_data_replay(
    model_type, data_root, mode, seed, n_samples,
    scaling_mode="standard", replay_ratio=0.2
):
    """
    Loads data for training from the UNIFORM generation pipeline.
    If mode='multi_stage', it manages an Experience Replay Buffer.
    
    NET2NET MODIFICATION:
    It enforces ANCHOR NORMALIZATION. The Mean/Std are calculated ONCE (Stage 1)
    and applied to all subsequent stages to ensure consistent physical scaling.
    """
    results = {}
    
    # 1. Define Stage Order based on Mode (Matches generateUniformMaxwell.py)
    if mode == "single_stage":
        stage_order = ["1.0_2.4"]
    elif mode == "multi_stage":
        stage_order = [
            "1.0_1.8", "1.8_2.0", "2.0_2.2", "2.2_2.4"
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Determine Device Name for Logging (CPU vs GPU)
    device_name = "CPU"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)

    # --- ANCHOR STORAGE ---
    # We will compute stats on the FIRST stage and lock them here.
    anchor_stats = {
        "X_mean": None, "X_std": None,
        "Y_mean": None, "Y_std": None
    }

    # 2. Iterate over stages
    for idx, stage_tag in enumerate(stage_order):
        t0 = time.time()
        
        # --- HEADER LOGGING ---
        print(f"\n=== [Net2Net Loader] Processing Stage: {stage_tag} (Ratio={replay_ratio}) ===")
        
        # --- PATH CONSTRUCTION ---
        if n_samples >= 1000:
            size_folder = f"{int(n_samples/1000)}ksamples"
        else:
            size_folder = f"{n_samples}samples"

        stage_dir = os.path.join(data_root, "uniform", mode, f"seed_{seed}", stage_tag, size_folder)
        # Standard filename structure from generateUniformMaxwell: X_3D_{model}_stage.pt
        X_path = os.path.join(stage_dir, f"X_3D_{model_type}_stage.pt")
        Y_path = os.path.join(stage_dir, f"Y_3D_{model_type}_stage.pt")
        
        # Prepare Directory for Images (Normalization & Replay Plots)
        norm_dir = os.path.join("images", "uniform", mode, f"seed_{seed}", stage_tag, size_folder, model_type, "normalized")  
        os.makedirs(norm_dir, exist_ok=True)

        print(f"✅ Loading data from: {stage_dir}")

        # Basic File Check
        if not (os.path.exists(X_path) and os.path.exists(Y_path)):
            print(f"[WARN] Missing files for stage {stage_tag} at {stage_dir}, skipping...")
            continue

        # Load Pytorch Tensors
        X = torch.load(X_path)
        Y = torch.load(Y_path)

        # 3. Experience Replay Logic (Active only in Multi-stage > Stage 1)
        if mode == "multi_stage" and idx > 0 and replay_ratio > 0:
            current_idx = idx
            num_current_samples = X.shape[0]
            num_replay = int(num_current_samples * replay_ratio)

            print(f"   ↺ Adding {num_replay} replay samples from previous stages...")
            replay_X_list, replay_Y_list = [], []
            
            # Retrieve data from all previous stages
            for prev_stage in stage_order[:current_idx]:
                prev_dir = os.path.join(data_root, "uniform", mode, f"seed_{seed}", prev_stage, size_folder)
                pX_path = os.path.join(prev_dir, f"X_3D_{model_type}_stage.pt")
                pY_path = os.path.join(prev_dir, f"Y_3D_{model_type}_stage.pt")
                
                if os.path.exists(pX_path) and os.path.exists(pY_path):
                    replay_X_list.append(torch.load(pX_path))
                    replay_Y_list.append(torch.load(pY_path))

            # Sample and Merge
            if replay_X_list:
                all_prev_X = torch.cat(replay_X_list, dim=0)
                all_prev_Y = torch.cat(replay_Y_list, dim=0)
                
                # Randomly sample 'num_replay' items from history
                if all_prev_X.shape[0] >= num_replay:
                    idxs = torch.randperm(all_prev_X.shape[0])[:num_replay]
                    X_replay = all_prev_X[idxs]
                    Y_replay = all_prev_Y[idxs]
                else:
                    X_replay = all_prev_X
                    Y_replay = all_prev_Y

                # --- VISUALIZATION STEP ---
                try:
                    print("   🎨 Generating Replay Invariant Diagram...")
                    plot_invariants_replay_mode(
                        X_curr=X.numpy(), 
                        X_replay=X_replay.numpy(), 
                        stage_tag=stage_tag, 
                        save_dir=norm_dir, 
                        model_name=model_type
                    )
                except Exception as e:
                    print(f"   ⚠️ Could not plot invariants: {e}")

                # Actual Merge for Training
                X = torch.cat([X, X_replay], dim=0)
                Y = torch.cat([Y, Y_replay], dim=0)
                print(f"   ✅ Combined dataset: {num_current_samples} (new) + {X_replay.shape[0]} (replay)")

        # Convert to NumPy for splitting operations
        X = X.numpy()
        Y = Y.numpy()

        # 4. Split Train/Val/Test
        # Strategy: 20% held out for Test. Remaining 80% split into Train/Val.
        # Deterministic Split: random_state=seed ensures index 0 is always Train
        X_train_val, X_test, Y_train_val, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=seed, shuffle=True
        )
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_train_val, Y_train_val, test_size=0.25, random_state=seed 
        )

        # 5. ANCHOR NORMALIZATION (The Key Logic)
        # If Stage 1 (idx 0): Calculate and lock the stats.
        if idx == 0:
            print("   ⚓ ESTABLISHING ANCHOR STATISTICS (Stage 1)")
            
            # Calculate on TRAIN set only
            anchor_stats["X_mean"] = X_train.mean(axis=0)
            anchor_stats["X_std"]  = X_train.std(axis=0)
            anchor_stats["X_std"][anchor_stats["X_std"] == 0] = 1.0 # Safety

            if scaling_mode == "standard":
                anchor_stats["Y_mean"] = Y_train.mean(axis=0)
                anchor_stats["Y_std"]  = Y_train.std(axis=0)
                anchor_stats["Y_std"][anchor_stats["Y_std"] == 0] = 1.0
            elif scaling_mode == "minmax":
                Y_min = Y_train.min(axis=0)
                Y_max = Y_train.max(axis=0)
                Y_range = np.where((Y_max - Y_min) == 0, 1.0, Y_max - Y_min)
                anchor_stats["Y_mean"] = Y_min
                anchor_stats["Y_std"] = Y_range
            else:
                raise ValueError(f"Invalid scaling_mode: {scaling_mode}")
            
            # Save Anchor stats to text file for verification
            with open(os.path.join(norm_dir, "ANCHOR_STATS.txt"), "w") as f:
                f.write(f"This file records the fixed normalization stats used for ALL stages.\n")
                f.write(f"Scaling Mode: {scaling_mode}\n\n")
                f.write(f"X_mean (L):\n{anchor_stats['X_mean']}\n\n")
                f.write(f"X_std (L):\n{anchor_stats['X_std']}\n\n")
                f.write(f"Y_mean (T):\n{anchor_stats['Y_mean']}\n\n")
                f.write(f"Y_std (T):\n{anchor_stats['Y_std']}\n\n")
            print(f"   ⚓ Anchors saved to {os.path.join(norm_dir, 'ANCHOR_STATS.txt')}")
        
        else:
            print("   ⚓ Using Pre-Calculated Anchor Statistics (Locked to Stage 1)")

        # Apply Normalization using Anchor Stats (For ALL stages)
        X_train_n = (X_train - anchor_stats["X_mean"]) / anchor_stats["X_std"]
        X_val_n   = (X_val   - anchor_stats["X_mean"]) / anchor_stats["X_std"]
        X_test_n  = (X_test  - anchor_stats["X_mean"]) / anchor_stats["X_std"]

        Y_train_n = (Y_train - anchor_stats["Y_mean"]) / anchor_stats["Y_std"]
        Y_val_n   = (Y_val   - anchor_stats["Y_mean"]) / anchor_stats["Y_std"]
        Y_test_n  = (Y_test  - anchor_stats["Y_mean"]) / anchor_stats["Y_std"]

        dt = time.time() - t0

        # --- DATA INTEGRITY LOGGING ---
        print(f"✅ Splitting Strategy: {scaling_mode} (Anchored)")
        print("✅ Data Integrity Check:")
        print(f"    • Input Shape (X): {X.shape}")
        print(f"    • Target Shape (Y): {Y.shape}")
        print(f"   ⏱ Stage splitting Time: {dt:.2f}s on {device_name}")

        # 6. Save Statistics to File (Original Detail)
        def write_stats(f, arr, label):
            """Helper to write stats to the text file."""
            flat = arr.flatten()
            mean_val, std_val = np.mean(arr), np.std(arr)
            min_val, max_val = np.min(arr), np.max(arr)
            q1_val, med_val, q3_val = np.percentile(arr, [25, 50, 75])
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

        with open(os.path.join(norm_dir, "normalizeddatastat.txt"), "w") as f:
            f.write(f"=== Stats for {model_type} Stage {stage_tag} (Uniform Replay) ===\n")
            f.write(f"NOTE: Data is normalized using Stage 1 ANCHOR stats.\n\n")
            write_stats(f, X_train_n, "Velocity Gradient (L) - Train Norm")
            write_stats(f, X_val_n,   "Velocity Gradient (L) - Val Norm")
            write_stats(f, X_test_n,  "Velocity Gradient (L) - Test Norm")
            write_stats(f, Y_train_n, "Stress Tensor (T) - Train Norm")

        # 7. Plot Histograms (Normalized Data Checks - Original Detail)
        def plot_hist_improved(X_data, Y_data, set_name):
            plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
            clean_model = model_type.replace("_", " ").title()

            # Helper for stats box
            def add_stats_box(ax, data, color):
                mean_val, std_val = np.mean(data), np.std(data)
                text_str = '\n'.join((r'$\mu=%.3f$' % mean_val, r'$\sigma=%.3f$' % std_val))
                props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=color)
                ax.text(0.95, 0.95, text_str, transform=ax.transAxes, fontsize=11,
                        verticalalignment='top', horizontalalignment='right', bbox=props)

            # Helper for info box
            def add_info_box(ax, set_label):
                text_str = f"Model: {clean_model}\nStage: {stage_tag}\nSet: {set_label}\n(Uniform-Anchored)"
                props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
                ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', horizontalalignment='left', bbox=props)

            # Plot L (Velocity Gradient)
            axes[0].hist(X_data.flatten(), bins=60, color='#4682B4', 
                         edgecolor='black', linewidth=0.5, log=True)
            axes[0].set_title(fr"Normalized $\mathbf{{L}}$ ")
            axes[0].grid(True, which="both", ls="--", alpha=0.3)
            axes[0].set_xlabel(r"Component Value ($L_{ij}$)", fontsize=14)
            axes[0].set_ylabel("Frequency", fontsize=14)
            add_stats_box(axes[0], X_data.flatten(), '#4682B4')
            add_info_box(axes[0], set_name)

            # Plot T (Stress Tensor)
            axes[1].hist(Y_data.flatten(), bins=60, color='#FF8C00', 
                         edgecolor='black', linewidth=0.5, log=True)
            axes[1].set_title(fr"Normalized $\mathbf{{T}}$")
            axes[1].grid(True, which="both", ls="--", alpha=0.3)
            axes[1].set_xlabel(r"Component Value ($T_{ij}$)", fontsize=14)
            axes[1].set_ylabel("Frequency", fontsize=14)
            add_stats_box(axes[1], Y_data.flatten(), '#FF8C00')
            add_info_box(axes[1], set_name)

            plt.savefig(os.path.join(norm_dir, f"hist_{set_name}.png"), dpi=300)
            plt.close()

        # Generate Histograms
        plot_hist_improved(X_train_n, Y_train_n, "Train")
        plot_hist_improved(X_val_n, Y_val_n, "Validation")
        plot_hist_improved(X_test_n, Y_test_n, "Test")

        # 8. Store Results (JAX Arrays for Training)
        # CRITICAL: We return the ANCHOR stats, not the local stage stats.
        results[stage_tag] = (
            jnp.array(X_train_n), jnp.array(X_val_n), jnp.array(X_test_n),
            jnp.array(Y_train_n), jnp.array(Y_val_n), jnp.array(Y_test_n),
            anchor_stats["X_mean"], anchor_stats["X_std"], 
            anchor_stats["Y_mean"], anchor_stats["Y_std"]
        )
        
    return results

# ===========================================================
# 4. Main Entry (Test Run)
# ===========================================================
@hydra.main(config_path="../config/data", config_name="dataConfig1", version_base=None)
def main(cfg: DictConfig):
    # This runs the loader in isolation to verify paths, stats, and plots.
    load_and_normalize_stagewise_data_replay(
        model_type=cfg.constitutive_eq,
        data_root="datafiles",
        mode=cfg.mode,
        seed=cfg.seed,
        n_samples=cfg.n_samples,
        scaling_mode="standard",
        replay_ratio=0.2
    )

if __name__ == "__main__":
    main()