import os
import numpy as np
import torch
import hydra
import time
import GPUtil
import random
from omegaconf import DictConfig
from scipy.linalg import solve_sylvester, LinAlgError

# =========================================================================
# Custom Utility Imports (PRESERVED from Original)
# =========================================================================
from utils.tensors import flatten_symmetric_tensors
from utils.flow_utils import generate_flow_L
from utils.write_flow_analysis import write_flow_analysis_file, plot_merged_flow_diagram
from utils.invariantsnew import filter_admissible_region

# =========================================================================
# Reproducibility Setup
# =========================================================================
def set_all_seeds(seed=42):
    """
    Set all possible random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================================================================
# Helper: Binning Logic (Condition Number -> Stage Tag)
# =========================================================================
def get_stage_tag(cond_val, mode="single_stage"):
    """
    Classifies a condition number into a specific folder/stage tag based on
    the user-defined structure.
    """
    # 1. Single Stage: One giant bin [1.0, 2.4]
    if mode == "single_stage":
        if 1.0 <= cond_val <= 2.4:
            return "1.0_2.4"
    
    # 2. Multi Stage: Granular bins
    elif mode == "multi_stage":
        
        # MERGED LOGIC:
        # We capture 1.0 (perfect stability) AND the range up to 1.2 in one bin.
        # We start at 0.99 to safely catch float precision issues (like 0.999999)
        if 0.99 <= cond_val < 1.2: return "1.0_1.2"
        
        # Subsequent Stages
        if 1.2 <= cond_val < 1.4: return "1.2_1.4"
        if 1.4 <= cond_val < 1.6: return "1.4_1.6"
        if 1.6 <= cond_val < 1.8: return "1.6_1.8"
        if 1.8 <= cond_val < 2.0: return "1.8_2.0"
        if 2.0 <= cond_val < 2.2: return "2.0_2.2"
        if 2.2 <= cond_val <= 2.4: return "2.2_2.4"

    return None

# =========================================================================
# Main Data Generation Routine (MODIFIED FOR MINING)
# =========================================================================
@hydra.main(config_path="config/data", config_name="flowConfig", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main Execution Pipeline for Maxwell-B Flow Data.
    
    MODIFICATIONS FOR MINING:
    1. Added 'target_min' and 'target_max' filtering logic.
    2. Changed output directory to 'maxwellflow'.
    3. Added 'Pre-Check' to skip solver if Condition Number is wrong.
    """

    # ---------------------------------------------------------------------
    # 0. Mining Configuration (New Logic)
    # ---------------------------------------------------------------------
    # Default behavior: Accept everything in standard range
    target_min = 1.0
    target_max = 2.4
    
    # Overwrite if user provided custom targets in command line
    # Usage: ++custom_min=2.0 ++custom_max=2.4
    if "custom_min" in cfg:
        target_min = float(cfg.custom_min)
        print(f"🎯 MINING MODE: Filtering for Condition Number >= {target_min}")
    if "custom_max" in cfg:
        target_max = float(cfg.custom_max)
        print(f"🎯 MINING MODE: Filtering for Condition Number < {target_max}")

    # ---------------------------------------------------------------------
    # 1. Define Bin Keys based on Config Mode
    # ---------------------------------------------------------------------
    if cfg.mode == "single_stage":
        bin_keys = ["1.0_2.4"]
        folder_mode = "single_stage"
        
    elif cfg.mode == "multi_stage":
        # These keys must match the output of get_stage_tag
        bin_keys = [
            "1.0_1.2",      # Ranges...
            "1.2_1.4",
            "1.4_1.6",
            "1.6_1.8",
            "1.8_2.0",
            "2.0_2.2",
            "2.2_2.4"
        ]
        folder_mode = "multi_stage"
    else:
        raise ValueError(f"Unknown cfg.mode: {cfg.mode}")

    # ---------------------------------------------------------------------
    # 2. Outer Loop: Iterate over Flow Types
    # ---------------------------------------------------------------------
    for flow_type in cfg.flow_types:
        print(f"\n=========================================================")
        print(f"🌊 STARTING FLOW TYPE: {flow_type.upper()}")
        print(f"=========================================================")

        flow_start = time.time()
        set_all_seeds(cfg.seed) # Reset seed for consistency

        # Initialize Bins: dictionary mapping 'tag' -> list of samples
        # Each entry stores tuple: (L, T, cond, resid)
        data_bins = {k: [] for k in bin_keys}
        
        total_collected = 0
        attempts = 0
        
        # MINING TWEAK: Increase max_attempts significantly.
        # Finding high-Wi samples at low rates is rare, so we need millions of tries.
        max_attempts = cfg.n_samples * 2000  

        print(f"   -> Mining for {cfg.n_samples} candidates (Range: {target_min}-{target_max})...")

        # -----------------------------------------------------------------
        # 3. Generation Loop (Global Pool)
        # -----------------------------------------------------------------
        while total_collected < cfg.n_samples and attempts < max_attempts:
            attempts += 1
            
            # A. Sample Rate 
            rate = np.random.uniform(cfg.rate_min, cfg.rate_max)
            
            # B. Generate L
            L = generate_flow_L(flow_type, rate, cfg.dim, cfg.max_vorticity_ratio)

            # C. Check Stability (Condition Number)
            A = np.eye(cfg.dim) - cfg.lam * L
            cA = np.linalg.cond(A)

            # --- MINING FILTER (Critical Optimization) ---
            # If the condition number isn't what we want, skip the expensive physics solver.
            if not (target_min <= cA < target_max):
                continue
            # ---------------------------------------------

            # D. Binning (Classify Sample)
            tag = get_stage_tag(cA, cfg.mode)
            
            # Only proceed if the sample falls into one of our defined bins
            if tag in data_bins:
                
                # E. Solve Physics
                B_mat = -cfg.lam * L.T
                D = 0.5 * (L + L.T)
                C_mat = 2.0 * cfg.eta0 * D
                
                try:
                    T = solve_sylvester(A, B_mat, C_mat)
                    
                    # Residual Check (Solver Accuracy)
                    Res = A @ T + T @ B_mat - C_mat
                    resid = np.linalg.norm(Res, 'fro')
                    
                    # Symmetry Check (Physics Consistency)
                    sym_err = np.linalg.norm(T - T.T, 'fro')
                    
                    # STRICT CHECK: 
                    if resid < 1e-12 and sym_err < 1e-12:
                        
                        # Force exact symmetry to remove 1e-16 noise
                        T = 0.5 * (T + T.T)

                        # Success: Add to bin
                        data_bins[tag].append((L, T, cA, resid))
                        total_collected += 1
                        
                        # Feedback for long mining runs
                        if total_collected % 500 == 0:
                            print(f"      Found {total_collected}/{cfg.n_samples} samples... (Attempts: {attempts})")
                        
                except (LinAlgError, ValueError):
                    pass

        print(f"   ✅ Generation finished. Candidates collected: {total_collected}")
        print(f"      Total Attempts: {attempts}")

        # -----------------------------------------------------------------
        # 4. Save & Analyze Each Bin
        # -----------------------------------------------------------------
        for stage_tag, items in data_bins.items():
            
            # Skip empty bins
            if len(items) == 0:
                continue

            print(f"   -> Processing Bin {stage_tag}: {len(items)} candidates")

            # Unzip Data
            L_list = [x[0] for x in items]
            T_list = [x[1] for x in items]
            condA_list = [x[2] for x in items]
            residual_list = [x[3] for x in items]

            # F. Filter Admissible Region (Physics Check - Lumley Triangle)
            try:
                filtered_L0, kept_mask = filter_admissible_region(L_list)
                
                # Apply mask to all parallel lists
                Y_final = [t for t, k in zip(T_list, kept_mask) if k]
                cond_final = [c for c, k in zip(condA_list, kept_mask) if k]
                res_final = [r for r, k in zip(residual_list, kept_mask) if k]
                L_final = filtered_L0
                
            except Exception as e:
                print(f"      ⚠️ Warning: Filtering failed for {stage_tag}: {e}")
                continue

            # Only save if we have valid samples remaining
            if len(L_final) > 0:
                
                # =========================================================
                # DEFINE SIZE FOLDER
                # =========================================================
                if cfg.n_samples >= 1000:
                    size_folder = f"{int(cfg.n_samples/1000)}ksamples"
                else:
                    size_folder = f"{cfg.n_samples}samples"

                # G. Construct Paths (TARGETING NEW FOLDER 'maxwellflow')
                # -------------------------------------------------------------
                # CHANGED: 'flowdata' -> 'maxwellflow' to separate mining output
                base_suffix = os.path.join("maxwellflow", folder_mode, f"seed_{cfg.seed}", stage_tag, flow_type, size_folder)
                # -------------------------------------------------------------
                
                base_data = os.path.join(cfg.paths.data, base_suffix)
                base_imgs = os.path.join(cfg.paths.images, base_suffix)
                
                os.makedirs(base_data, exist_ok=True)
                os.makedirs(base_imgs, exist_ok=True)

                # H. Save Tensors (.pt)
                X_np = np.array(L_final)
                Y_np = np.array(Y_final)
                
                X_flat = X_np.reshape(X_np.shape[0], -1)
                Y_flat = flatten_symmetric_tensors(Y_np)
                
                suffix = "_stage"
                torch.save(torch.tensor(X_flat, dtype=torch.float32), 
                           os.path.join(base_data, f"X_{cfg.dim}D_{flow_type}{suffix}.pt"))
                torch.save(torch.tensor(Y_flat, dtype=torch.float32), 
                           os.path.join(base_data, f"Y_{cfg.dim}D_{flow_type}{suffix}.pt"))

                print(f"      ✅ Saved {len(L_final)} samples to {base_data}")

                # I. Generate Analysis Plots
                try:
                    write_flow_analysis_file(
                        model_name="maxwell_B",
                        flow_type=flow_type,
                        stability_status=cfg.mode,
                        L_list=L_final,
                        T_list=Y_final,
                        condA_list=cond_final,
                        lam=cfg.lam,
                        save_root=base_imgs,
                        cfg=cfg,
                        X_flat=X_flat,
                        Y_flat=Y_flat,
                        stage_tag=stage_tag,
                        residual_list=res_final
                    )
                except Exception as e:
                    print(f"      ⚠️ Plotting error in {stage_tag}: {e}")
            else:
                print(f"      ⚠️ Bin {stage_tag} empty after Admissible Region filter.")

        elapsed = time.time() - flow_start
        gpus = GPUtil.getGPUs()
        device_name = gpus[0].name if gpus else "CPU"
        print(f"      ⏱ Flow Type Time: {elapsed:.2f}s on {device_name}")

    # End of Flow Types Loop

    # ---------------------------------------------------------------------
    # 5. Generate Merged Summary Plots (All Flows Combined)
    # ---------------------------------------------------------------------
    print("\n=========================================================")
    print("🔄 GENERATING MERGED INVARIANT PLOTS")
    print("=========================================================")
    
    # Iterate through all bins (including 1.0) to generate summaries
    for st in bin_keys:
        # Define root folder for this stage (TARGETING NEW FOLDER 'maxwellflow')
        data_root_stage = os.path.join(cfg.paths.data, "maxwellflow", folder_mode, f"seed_{cfg.seed}", st)
        images_root_stage = os.path.join(cfg.paths.images, "maxwellflow", folder_mode, f"seed_{cfg.seed}", st)
        
        # Check if this stage folder exists
        if os.path.exists(data_root_stage):
            try:
                plot_merged_flow_diagram(data_root_stage, images_root_stage, st)
            except Exception as e:
                print(f"   ⚠️ Failed to merge plots for {st}: {e}")
        else:
            # This is normal for empty bins
            print(f"   ⚠️ Skipping merge for {st} (No data generated)")

    print("\n✅ Flow Data Generation & Merging Complete.")

if __name__ == "__main__":
    main()