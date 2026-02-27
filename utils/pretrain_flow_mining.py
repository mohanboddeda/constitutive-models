#!/usr/bin/env python3
"""
Pre-training Data Loader & Statistics Generator (Flow Types) - MINING EDITION
Responsible for: 
1. Loading Stage-wise Data (Single/Multi-stage) for specific Flow Types.
2. Managing Experience Replay Buffer (preventing catastrophic forgetting).
3. Normalizing Data (Standard/MinMax) - ANCHORED STRATEGY.
   - Calculates Mean/Std on Stage 1 (Base Physics).
   - Applies these fixed stats to all subsequent stages to preserve magnitude changes.
4. Visualizing Data Distributions and Invariant Space (Lumley Triangle).

ADAPTATION:
- Supports 'Inverse Pyramid' strategy where sample sizes differ per stage.
- Automatically detects the correct sample folder for Replay Data.
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
# 1. Helper: Find Variable Sample Folders (MINING SPECIFIC)
# ===========================================================
def find_correct_size_folder(data_root, mode, seed, stage, flow_type, target_n=None):
    """
    Scans the directory to find the actual size folder.
    STRICT FIX: Prioritizes the EXACT match for 'target_n' (e.g. 10ksamples).
    Does NOT default to larger folders if the exact one exists.
    """
    base_path = os.path.join(data_root, "maxwellflow", mode, f"seed_{seed}", stage, flow_type)
    
    if not os.path.exists(base_path):
        return None
        
    # Get all subdirectories that contain "samples"
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and "samples" in d]
    
    if not subdirs:
        return None

    # --- Strategy 1: Strict Exact Match (Highest Priority) ---
    if target_n is not None:
        # Check for '10ksamples' format (e.g. 10000 -> 10ksamples)
        k_val = int(target_n / 1000)
        k_format = f"{k_val}ksamples"
        
        # Check for '10000samples' format
        raw_format = f"{target_n}samples"
        
        # If we find the EXACT folder, return it immediately.
        if k_format in subdirs:
            return k_format
        if raw_format in subdirs:
            return raw_format

    # --- Strategy 2: Fallback (Smallest Size First) ---
    # Parse folder names to integers: "10ksamples" -> 10000
    def parse_size(folder_name):
        clean = folder_name.lower().replace("samples", "").replace("k", "000")
        try:
            return int(clean)
        except:
            return 999999999 # Push weird names to the end

    # Sort available folders by size (Small -> Large)
    # This prevents '100ksamples' from being picked before '10ksamples'
    subdirs.sort(key=parse_size)

    # Return the smallest valid folder found
    return subdirs[0]
# ===========================================================
# 2. Helper: Invariant Space Visualization (Replay Mode)
# ===========================================================
def plot_invariants_replay_mode(X_curr, X_replay, stage_tag, save_dir, flow_type):
    """
    Plots the invariant space (II vs III).
    - Stage 1: Shows ONLY current samples (Red).
    - Stage 1.2+: Shows current (Red) + Replay (Blue).
    """
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    
    # --- Helper to calculate invariants ---
    def get_invariants(X_data):
        if X_data.ndim == 2:
            X_data = X_data.reshape(-1, 3, 3)
        D = 0.5 * (X_data + np.swapaxes(X_data, 1, 2))
        _, II, III = compute_invariants_vectorized(D)
        return -II, III

    # 1. Get Current Stage Data
    II_curr, III_curr = get_invariants(X_curr)
    
    # 2. Check for Replay Data (Strict Check)
    has_replay = (X_replay is not None and len(X_replay) > 0)
    
    if has_replay:
        II_repl, III_repl = get_invariants(X_replay)

    # --- 3. Background (Lumley Triangle) ---
    II_vals = np.linspace(0, 1.6, 400)
    III_vals = np.linspace(-0.7, 0.7, 400) 
    II_grid, III_grid = np.meshgrid(II_vals, III_vals)
    
    discriminant = ((-III_grid / 2)**2 + (-II_grid / 3)**3)

    # Colors: Admissible (Pink), Inadmissible (Blue) - Matching First Case
    plt.contourf(II_grid, III_grid, discriminant, levels=[-1e10, 0, 1e10],
                 colors=['#ffa8a8', '#a8c6ff'], alpha=0.8, zorder=0)

    # Boundaries
    x_b = np.linspace(0, 1.6, 1000)
    y_b = 2 * np.sqrt((x_b/3)**3)
    plt.plot(x_b, y_b, 'k-', lw=1.5, zorder=1)
    plt.plot(x_b, -y_b, 'k-', lw=1.5, zorder=1)
    plt.axhline(0, color='k', lw=1.5, zorder=1)

    # --- 4. Scatter Points ---
    # Current Stage = RED
    plt.scatter(II_curr, III_curr, c='red', s=30, alpha=0.6, 
                edgecolors='k', linewidth=0.5, zorder=2)
    
    # Replay Buffer = BLUE (Only if it exists)
    if has_replay:
        plt.scatter(II_repl, III_repl, c='blue', s=30, alpha=0.6, 
                    edgecolors='k', linewidth=0.5, zorder=3)

    # --- 5. Formatting & Legend ---
    clean_flow = flow_type.replace("_", " ").title()
    plt.title(f"Invariant Space: Replay Mode (Stage {stage_tag})", fontsize=16, pad=10)
    plt.xlabel(r"Second Invariant ($-II_{\mathbf{D}}$)", fontsize=14)
    plt.ylabel(r"Third Invariant ($III_{\mathbf{D}}$)", fontsize=14)
    plt.xlim(0, 1.5)
    plt.ylim(-0.6, 0.6)

    n_curr = len(X_curr)
    
    # Build Legend
    legend_elements = [
        Patch(facecolor='#ffa8a8', edgecolor='none', label='Admissible Region'),
        Line2D([0], [0], marker='o', color='w', label=f'Current Stage (n={n_curr})',
               markerfacecolor='red', markersize=8, markeredgecolor='k'),
    ]

    # ONLY add Replay to legend if we actually plotted it
    if has_replay:
        n_replay = len(X_replay)
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', label=f'Replay Buffer (n={n_replay})',
                   markerfacecolor='blue', markersize=8, markeredgecolor='k')
        )
        
    legend_elements.append(Line2D([0], [0], color='none', label=f'Flow: {clean_flow}'))
    
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.95, 
              edgecolor='#cccccc', fontsize=10)
    
    plt.grid(True, linestyle='--', alpha=0.4, zorder=0)
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"Replay_Invariants_{stage_tag}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"   📊 Replay Invariant Plot saved: {out_path}")

# ===========================================================
# 3. Stage-wise Replay Loader (Flow Logic)
# ===========================================================
def load_and_normalize_flow_data_replay_mining(
    flow_type, model_type, data_root, mode, seed, n_samples,
    scaling_mode="standard", replay_ratio=0.2
):
    """
    Loads data for training from the FLOW generation pipeline.
    If mode='multi_stage', it manages an Experience Replay Buffer.
    It calculates normalization stats on the TRAIN split only.
    
    CRITICAL CHANGE: Uses 'maxwellflow' folder and Anchored Normalization.
    """
    results = {}
    
    # 1. Define Stage Order based on Mode (Matches generateFlowMaxwell.py)
    if mode == "single_stage":
        stage_order = ["1.0_2.4"]
    elif mode == "multi_stage":
        stage_order = [
            "1.0_1.2", "1.2_1.4", "1.4_1.6", 
            "1.6_1.8", "1.8_2.0", "2.0_2.2", "2.2_2.4"
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Determine Device Name for Logging (CPU vs GPU)
    device_name = "CPU"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)

    # --- NEW: CONTAINER TO FREEZE STATISTICS (Anchored Normalization) ---
    frozen_stats = {} 
    # --------------------------------------------------------------------

    # 2. Iterate over stages
    for stage_tag in stage_order:
        t0 = time.time()
        
        # --- HEADER LOGGING ---
        if mode == "single_stage":
            print(f"\n=== Single Stage Training: {flow_type} ===")
        else:
            print(f"\n=== Stage {stage_tag} ({flow_type}) - MINING LOADER (Ratio={replay_ratio}) ===")
        
        # --- PATH CONSTRUCTION (Smart Mining Logic) ---
        # 1. Try to guess the folder name based on current n_samples argument
        if n_samples >= 1000:
            size_folder = f"{int(n_samples/1000)}ksamples"
        else:
            size_folder = f"{n_samples}samples"

        # 2. Check if this exact folder exists in 'maxwellflow'
        stage_dir = os.path.join(data_root, "maxwellflow", mode, f"seed_{seed}", stage_tag, flow_type, size_folder)
        
        # 3. If not found, use auto-detection (Smart Fallback)
        if not os.path.exists(stage_dir):
             # FIX 1: Pass target_n=n_samples to ensure strict matching
             detected_size = find_correct_size_folder(data_root, mode, seed, stage_tag, flow_type, target_n=n_samples)
             
             if detected_size:
                 # Only print switching warning if it's actually different
                 if detected_size != size_folder:
                     print(f"   ⚠️ Requested {size_folder} but found {detected_size}. Switching...")
                 size_folder = detected_size
                 stage_dir = os.path.join(data_root, "maxwellflow", mode, f"seed_{seed}", stage_tag, flow_type, size_folder)
             else:
                 # If truly missing, skip
                 print(f"[WARN] Missing files for {flow_type} at Stage {stage_tag} (Path: {stage_dir})")
                 print("       Skipping this stage...")
                 continue
        
        # File Names: X_3D_<flow_type>_stage.pt
        X_path = os.path.join(stage_dir, f"X_3D_{flow_type}_stage.pt")
        Y_path = os.path.join(stage_dir, f"Y_3D_{flow_type}_stage.pt")
        
        # Prepare Directory for Images (Normalization & Replay Plots)
        norm_dir = os.path.join("images", "maxwellflow", mode, f"seed_{seed}", stage_tag, flow_type, size_folder, "normalized")  
        os.makedirs(norm_dir, exist_ok=True)

        # Basic File Check (Crucial for Flow Data which might be sparse)
        if not (os.path.exists(X_path) and os.path.exists(Y_path)):
            print(f"[WARN] Missing files for {flow_type} at Stage {stage_tag} (Path: {stage_dir})")
            print("       Skipping this stage...")
            continue

        print(f"✅ Loading data from: {stage_dir}")

        # Load Pytorch Tensors
        X = torch.load(X_path)
        Y = torch.load(Y_path)

        # 3. Experience Replay Logic (Active only in Multi-stage > Stage 1)
        # We need a holder for visualization even if replay is empty
        X_replay_viz = np.array([]) 

        if mode == "multi_stage" and stage_tag in stage_order and stage_order.index(stage_tag) > 0 and replay_ratio > 0:
            current_idx = stage_order.index(stage_tag)
            num_current_samples = X.shape[0]
            num_replay = int(num_current_samples * replay_ratio)

            print(f"   ↺ Adding {num_replay} replay samples from previous stages...")
            replay_X_list, replay_Y_list = [], []
            
            # Retrieve data from all previous stages
            for prev_stage in stage_order[:current_idx]:
                
                # SMART REPLAY: Find the actual size folder for the previous stage
                # FIX 2: Pass target_n=n_samples here too
                prev_size_folder = find_correct_size_folder(data_root, mode, seed, prev_stage, flow_type, target_n=n_samples)
                
                if prev_size_folder:
                    prev_dir = os.path.join(data_root, "maxwellflow", mode, f"seed_{seed}", prev_stage, flow_type, prev_size_folder)
                    pX_path = os.path.join(prev_dir, f"X_3D_{flow_type}_stage.pt")
                    pY_path = os.path.join(prev_dir, f"Y_3D_{flow_type}_stage.pt")
                    
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
                
                # Keep a copy for plotting before merging
                X_replay_viz = X_replay.numpy() 

                # Actual Merge for Training
                X = torch.cat([X, X_replay], dim=0)
                Y = torch.cat([Y, Y_replay], dim=0)
                print(f"   ✅ Combined dataset: {num_current_samples} (new) + {X_replay.shape[0]} (replay)")

        # Convert to NumPy for splitting operations
        X = X.numpy()
        Y = Y.numpy()

        # --- VISUALIZATION STEP (FIXED) ---
        if mode == "multi_stage":
            # FIX 3: STRICT CHECK - Only plot if Replay Buffer actually has data!
            if len(X_replay_viz) > 0:
                try:
                    print("   🎨 Generating Replay Invariant Diagram...")
                    
                    # Slice X to separate current/replay for plotting
                    len_replay = len(X_replay_viz)
                    X_curr_plot = X[:-len_replay]
                        
                    plot_invariants_replay_mode(
                        X_curr=X_curr_plot, 
                        X_replay=X_replay_viz, 
                        stage_tag=stage_tag, 
                        save_dir=norm_dir, 
                        flow_type=flow_type
                    )
                except Exception as e:
                    print(f"   ⚠️ Could not plot invariants: {e}")
            else:
                # This ensures Stage 1 (empty replay) prints this instead of plotting empty
                print("   ⏩ Skipping Invariant Plot (No Replay Data in this stage).")

        # 4. Split Train/Val/Test
        # Strategy: 20% held out for Test. Remaining 80% split into Train/Val.
        if len(X) < 10:
            print("   ⚠️ Not enough samples to split. Skipping.")
            continue

        X_train_val, X_test, Y_train_val, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=seed, shuffle=True
        )
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_train_val, Y_train_val, test_size=0.25, random_state=seed # 0.25 of 0.8 is 0.2 total
        )

        # 5. Normalization (ANCHORED STRATEGY)
        # Logic: Calculate statistics on Stage 1 (or the first loaded stage). 
        # Freeze them and apply to all subsequent stages.
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
            if not frozen_stats:
                # Fallback if Stage 1 was skipped (e.g. file missing): Calculate fresh
                print(f"   ⚠️ [Normalization] Anchor stats missing (Stage 1 skipped). Calculating fresh stats...")
                X_mean = X_train.mean(axis=0)
                X_std = X_train.std(axis=0)
                X_std[X_std == 0] = 1.0
                Y_mean = Y_train.mean(axis=0)
                Y_std = Y_train.std(axis=0)
                Y_std[Y_std == 0] = 1.0
                frozen_stats = {'X_mean': X_mean, 'X_std': X_std, 'Y_mean': Y_mean, 'Y_std': Y_std}
            else:
                print(f"   🔓 [Normalization] Loading Anchor stats (Frozen from Stage 1)...")
                X_mean = frozen_stats['X_mean']
                X_std  = frozen_stats['X_std']
                Y_mean = frozen_stats['Y_mean']
                Y_std  = frozen_stats['Y_std']
        
        # Apply normalization (using the selected X_mean, X_std, etc.)
        X_train_n = (X_train - X_mean) / X_std
        X_val_n = (X_val - X_mean) / X_std
        X_test_n = (X_test - X_mean) / X_std

        Y_train_n = (Y_train - Y_mean) / Y_std
        Y_val_n = (Y_val - Y_mean) / Y_std
        Y_test_n = (Y_test - Y_mean) / Y_std

        dt = time.time() - t0

        # --- DATA INTEGRITY LOGGING ---
        print(f"✅ Splitting Strategy: {strategy_name} (Anchored)")
        print("✅ Data Integrity Check:")
        print(f"    • Input Shape (X): {X.shape}")
        print(f"    • Target Shape (Y): {Y.shape}")
        print(f"   ⏱ Stage splitting Time: {dt:.2f}s on {device_name}")
        print(f"✅ Pretraining Setup Complete for Stage {stage_tag}")

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
            f.write(f"=== Stats for {flow_type} Stage {stage_tag} (Flow Replay) ===\n")
            write_stats(f, X_train_n, "Velocity Gradient (L) - Train Norm")
            write_stats(f, X_val_n,   "Velocity Gradient (L) - Val Norm")
            write_stats(f, X_test_n,  "Velocity Gradient (L) - Test Norm")
            write_stats(f, Y_train_n, "Stress Tensor (T) - Train Norm")

        # 7. Plot Histograms (Normalized Data Checks)
        def plot_hist_improved(X_data, Y_data, set_name):
            plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
            clean_flow = flow_type.replace("_", " ").title()

            # Helper for stats box
            def add_stats_box(ax, data, color):
                mean_val, std_val = np.mean(data), np.std(data)
                text_str = '\n'.join((r'$\mu=%.3f$' % mean_val, r'$\sigma=%.3f$' % std_val))
                props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=color)
                ax.text(0.95, 0.95, text_str, transform=ax.transAxes, fontsize=11,
                        verticalalignment='top', horizontalalignment='right', bbox=props)

            # Helper for info box
            def add_info_box(ax, set_label):
                text_str = f"Flow: {clean_flow}\nStage: {stage_tag}\nSet: {set_label}"
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
        f.write(serialization.msgpack_serialize(to_save))

def load_checkpoint(path, init_params=None):
    """
    Loads model parameters. 
    If init_params is None, returns raw dict (Crucial for Net2Net).
    """
    with open(path, "rb") as f:
        data = f.read()
    restored = serialization.msgpack_restore(data)
    if init_params is not None:
        return serialization.from_state_dict(init_params, restored)
    return restored

# ===========================================================
# 4. Main Entry (Test Run for ALL Flow Types)
# ===========================================================
@hydra.main(config_path="../config/data", config_name="flowConfig", version_base=None)
def main(cfg: DictConfig):
    # This runs the loader for ALL defined flow types in config
    for flow_type in cfg.flow_types:
        print(f"\n >>> PROCESSING FLOW TYPE: {flow_type} <<<")
        
        # We catch exceptions so one failed flow type doesn't crash the whole loop
        try:
            load_and_normalize_flow_data_replay_mining(
                flow_type=flow_type,
                model_type=cfg.constitutive_eq,
                data_root="datafiles",
                mode=cfg.mode,
                seed=cfg.seed,
                n_samples=cfg.n_samples,
                scaling_mode="standard",
                replay_ratio=0.2
            )
        except Exception as e:
            print(f"❌ Critical Error in Flow Type {flow_type}: {e}")

if __name__ == "__main__":
    main()