#!/usr/bin/env python3
"""
Pre-training Data Loader & Statistics Generator
Responsible for: 
1. Loading Stage-wise Data (Single/Multi-stage).
2. Managing Experience Replay Buffer (preventing catastrophic forgetting).
3. Normalizing Data (Standard/MinMax) without leakage.
4. Visualizing Data Distributions and Invariant Space (Lumley Triangle).
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

# --- FIX: Add project root to path so 'utils' module can be found ---
# This allows the script to be run from the root using: python utils/pretrain_random.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# --------------------------------------------------------------------

# Ensure this utility exists in your project structure
# If it's in a different path, adjust the import accordingly.
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
    
    # Updated Title as requested
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
        Line2D([0], [0], color='none', label=f'Model: {clean_model}'),
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
# 2. Stage-wise Replay Loader (Main Logic)
# ===========================================================
def load_and_normalize_stagewise_data_replay(
    model_type, data_root, mode, seed, n_samples,
    scaling_mode="standard", replay_ratio=0.2
):
    """
    Loads data for training. 
    If mode='multi_stage', it manages an Experience Replay Buffer.
    
    CRITICAL UPDATE: 
    - Calculates normalization statistics (Mean/Std) ONLY on the First Stage.
    - Freezes these 'Anchor Stats' and applies them to all subsequent stages.
    - Prevents Covariate Shift (divergence) in Curriculum Learning.
    """
    results = {}
    
    # 1. Define Stage Order based on Mode
    if mode == "single_stage":
        stage_order = ["1.0_2.4"]
    elif mode == "multi_stage":
        stage_order = [
            "1.0_1.2", "1.2_1.4", "1.4_1.6", "1.6_1.8", 
            "1.8_2.0", "2.0_2.2", "2.2_2.4"
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Determine Device Name for Logging (CPU vs GPU)
    device_name = "CPU"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)

    # --- NEW: CONTAINER TO FREEZE STATISTICS ---
    frozen_stats = {} 
    # -------------------------------------------

    # 2. Iterate over stages
    for stage_tag in stage_order:
        t0 = time.time()
        
        # --- HEADER LOGGING ---
        if mode == "single_stage":
            print(f"\n=== Single Stage Training: {model_type} ===")
        else:
            print(f"\n=== Stage {stage_tag} ({model_type}) - REPLAY MODE (Ratio={replay_ratio}) ===")
        
        # =========================================================
        # AUTO-DETECT LOGIC: Find largest available dataset folder
        # =========================================================
        stage_base_dir = os.path.join(data_root, "random", mode, f"seed_{seed}", stage_tag)
        
        # Default fallback (if folders missing)
        if n_samples >= 1000:
            size_folder = f"{int(n_samples/1000)}ksamples"
        else:
            size_folder = f"{n_samples}samples"

        if os.path.exists(stage_base_dir):
            # Find all subfolders like "10ksamples", "50ksamples"
            subdirs = [d for d in os.listdir(stage_base_dir) 
                       if os.path.isdir(os.path.join(stage_base_dir, d)) and "samples" in d]
            
            if subdirs:
                # Helper to parse "50ksamples" -> 50000
                def parse_size(name):
                    digits = "".join(filter(str.isdigit, name))
                    if not digits: return 0
                    val = int(digits)
                    return val * 1000 if "k" in name else val

                # Pick the folder with the MOST samples
                best_folder = max(subdirs, key=parse_size)
                size_folder = best_folder
                
                # Print confirmation so you know it worked
                n_found = parse_size(best_folder)
                print(f"   🔍 Auto-Detected best dataset: {size_folder} ({n_found} samples)")
# =========================================================

        # Construct File Paths with size_folder
        stage_dir = os.path.join(data_root, "random", mode, f"seed_{seed}", stage_tag, size_folder)
        X_path = os.path.join(stage_dir, f"X_3D_{model_type}_stage.pt")
        Y_path = os.path.join(stage_dir, f"Y_3D_{model_type}_stage.pt")
        
        # Prepare Directory for Images (Normalization & Replay Plots)
        norm_dir = os.path.join("images", "random", mode, f"seed_{seed}", stage_tag, size_folder, model_type, "normalized")  
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
        if mode == "multi_stage" and stage_tag in stage_order and stage_order.index(stage_tag) > 0 and replay_ratio > 0:
            current_idx = stage_order.index(stage_tag)
            num_current_samples = X.shape[0]
            num_replay = int(num_current_samples * replay_ratio)

            print(f"   ↺ Adding {num_replay} replay samples from previous stages...")
            replay_X_list, replay_Y_list = [], []
            
            # Retrieve data from all previous stages (With Auto-Detect Logic)
            for prev_stage in stage_order[:current_idx]:
                # 1. Base path to PREVIOUS stage
                prev_base_dir = os.path.join(data_root, "random", mode, f"seed_{seed}", prev_stage)
                
                # 2. AUTO-DETECT: Find the largest sample folder in that previous stage
                replay_size_folder = ""
                found_folder = False
                
                if os.path.exists(prev_base_dir):
                    subdirs = [d for d in os.listdir(prev_base_dir) 
                               if os.path.isdir(os.path.join(prev_base_dir, d)) and "samples" in d]
                    
                    if subdirs:
                        # Helper to parse "50ksamples" -> 50000
                        def parse_size(name):
                            digits = "".join(filter(str.isdigit, name))
                            if not digits: return 0
                            val = int(digits)
                            return val * 1000 if "k" in name else val

                        # Pick LARGEST folder (Pyramid Logic)
                        replay_size_folder = max(subdirs, key=parse_size)
                        found_folder = True
                
                # Fallback if detection fails (uses config n_samples - usually 10k default)
                if not found_folder:
                    if n_samples >= 1000: replay_size_folder = f"{int(n_samples/1000)}ksamples"
                    else: replay_size_folder = f"{n_samples}samples"

                # 3. Construct Final Path using the correct folder for THAT stage
                prev_dir = os.path.join(prev_base_dir, replay_size_folder)
                pX_path = os.path.join(prev_dir, f"X_3D_{model_type}_stage.pt")
                pY_path = os.path.join(prev_dir, f"Y_3D_{model_type}_stage.pt")
                
                # 4. Load Data
                if os.path.exists(pX_path) and os.path.exists(pY_path):
                    try:
                        replay_X_list.append(torch.load(pX_path))
                        replay_Y_list.append(torch.load(pY_path))
                    except Exception as e:
                        print(f"   ⚠️ Error loading replay from {prev_stage}: {e}")
                else:
                    # Optional debug: uncomment if needed
                    # print(f"   ⚠️ Replay file not found for {prev_stage}: {pX_path}")
                    pass

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
                    # plot_invariants_replay_mode is defined outside, ensuring it is called correctly if it exists
                    if 'plot_invariants_replay_mode' in globals():
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
        X_train_val, X_test, Y_train_val, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=seed, shuffle=True
        )
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_train_val, Y_train_val, test_size=0.25, random_state=seed # 0.25 of 0.8 is 0.2 total
        )

        # ===========================================================
        # 5. Normalization (ANCHOR STATS FIX)
        # ===========================================================
        is_first_stage = (stage_tag == stage_order[0])
        strategy_name = scaling_mode.title()

        # A. Calculate Stats ONLY on First Stage
        if is_first_stage:
            print(f"   🔒 [Normalization] Calculating GLOBAL Anchor stats from Stage {stage_tag}...")
            
            # Input Stats (X)
            X_mean = X_train.mean(axis=0)
            X_std = X_train.std(axis=0)
            X_std[X_std == 0] = 1.0 # Safety check
            
            # Output Stats (Y)
            if scaling_mode == "standard":
                Y_mean = Y_train.mean(axis=0)
                Y_std = Y_train.std(axis=0)
                Y_std[Y_std == 0] = 1.0
            elif scaling_mode == "minmax":
                Y_min = Y_train.min(axis=0)
                Y_max = Y_train.max(axis=0)
                Y_range = np.where((Y_max - Y_min) == 0, 1.0, Y_max - Y_min)
                Y_mean, Y_std = Y_min, Y_range # Map Min->Mean, Range->Std for consistent application
            else:
                raise ValueError(f"Invalid scaling_mode: {scaling_mode}")
            
            # Save to Frozen Container
            frozen_stats = {
                'X_mean': X_mean, 'X_std': X_std,
                'Y_mean': Y_mean, 'Y_std': Y_std
            }
        else:
            # B. Load Frozen Stats for Later Stages
            print(f"   🔓 [Normalization] Loading Anchor stats (Frozen from Stage 1)...")
            X_mean = frozen_stats['X_mean']
            X_std  = frozen_stats['X_std']
            Y_mean = frozen_stats['Y_mean']
            Y_std  = frozen_stats['Y_std']

        # C. Apply Normalization (Using the fixed stats)
        # Note: For MinMax, Y_mean is min and Y_std is range, so this formula works for both.
        X_train_n = (X_train - X_mean) / X_std
        X_val_n   = (X_val   - X_mean) / X_std
        X_test_n  = (X_test  - X_mean) / X_std

        Y_train_n = (Y_train - Y_mean) / Y_std
        Y_val_n   = (Y_val   - Y_mean) / Y_std
        Y_test_n  = (Y_test  - Y_mean) / Y_std
        # ===========================================================

        dt = time.time() - t0

        # --- DATA INTEGRITY LOGGING ---
        print(f"✅ Splitting Strategy: {strategy_name} (Anchored)")
        print("✅ Data Integrity Check:")
        print(f"    • Input Shape (X): {X.shape}")
        print(f"    • Target Shape (Y): {Y.shape}")
        print(f"   ⏱ Stage splitting Time: {dt:.2f}s on {device_name}")
        print(f"✅ Pretraining Setup Complete for {mode if mode == 'single_stage' else 'Stage'}={stage_tag}")

        # 6. Save Statistics to File
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
            f.write(f"=== Stats for {model_type} Stage {stage_tag} (Replay) ===\n")
            write_stats(f, X_train_n, "Velocity Gradient (L) - Train Norm")
            write_stats(f, X_val_n,   "Velocity Gradient (L) - Val Norm")
            write_stats(f, X_test_n,  "Velocity Gradient (L) - Test Norm")
            write_stats(f, Y_train_n, "Stress Tensor (T) - Train Norm")

        # 7. Plot Histograms (Normalized Data Checks)
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
                text_str = f"Model: {clean_model}\nStage: {stage_tag}\nSet: {set_label}"
                props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
                ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', horizontalalignment='left', bbox=props)

            # Plot L (Velocity Gradient)
            axes[0].hist(X_data.flatten(), bins=60, color='#4682B4', 
                         edgecolor='black', linewidth=0.5, log=True)
            axes[0].set_title(fr"Normalized $\mathbf{{L}}$ ")
            axes[0].grid(True, which="both", ls="--", alpha=0.3)
            add_stats_box(axes[0], X_data.flatten(), '#4682B4')
            add_info_box(axes[0], set_name)

            # Plot T (Stress Tensor)
            axes[1].hist(Y_data.flatten(), bins=60, color='#FF8C00', 
                         edgecolor='black', linewidth=0.5, log=True)
            axes[1].set_title(fr"Normalized $\mathbf{{T}}$")
            axes[1].grid(True, which="both", ls="--", alpha=0.3)
            add_stats_box(axes[1], Y_data.flatten(), '#FF8C00')
            add_info_box(axes[1], set_name)

            plt.savefig(os.path.join(norm_dir, f"hist_{set_name}.png"), dpi=300)
            plt.close()

        # Generate Histograms
        try:
            plot_hist_improved(X_train_n, Y_train_n, "Train")
            plot_hist_improved(X_val_n, Y_val_n, "Validation")
            plot_hist_improved(X_test_n, Y_test_n, "Test")
        except Exception as e:
            print(f"   ⚠️ Could not plot histograms: {e}")

        # 8. Store Results (JAX Arrays for Training)
        results[stage_tag] = (
            jnp.array(X_train_n), jnp.array(X_val_n), jnp.array(X_test_n),
            jnp.array(Y_train_n), jnp.array(Y_val_n), jnp.array(Y_test_n),
            X_mean, X_std, Y_mean, Y_std
        )
        
    return results

# ===========================================================
# 3. Checkpoint Utilities
# ===========================================================
def save_checkpoint(params, X_mean, X_std, Y_mean, Y_std, path):
    """Saves model parameters and normalization stats via Flax."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    to_save = {
        "params": params, 
        "X_mean": X_mean, "X_std": X_std, 
        "Y_mean": Y_mean, "Y_std": Y_std
    }
    with open(path, "wb") as f:
        f.write(flax.serialization.to_bytes(to_save))

def load_checkpoint(path, init_params):
    """Loads model parameters from bytes."""
    with open(path, "rb") as f:
        restored = flax.serialization.from_bytes(init_params, f.read())
    return restored

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