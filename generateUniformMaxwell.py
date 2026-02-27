import os
import numpy as np
import torch
import hydra
import time
import GPUtil
import random
from omegaconf import DictConfig

# --- IMPORTS ---
from scipy.linalg import solve_sylvester
from numpy.linalg import cond
from utils.tensors import generate_random_rotation_matrix, flatten_symmetric_tensors

# Uniform Utilities
from utils.uniform_utils import get_eigenvalues_from_invariants, generate_invariant_grid
from utils.write_uniform_analysis import write_uniform_analysis_file 
from utils.invariantsnew import filter_admissible_region
from utils.plot_stage_progression1 import plot_stage_progression

# =========================================================================
# Fix Seed for Reproducibility
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
# Main Hydra Entry Point
# =========================================================================
@hydra.main(config_path="config/data", config_name="dataConfig1", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main UNIFORM data generation routine for Maxwell-B.
    
    Uses a Structured Grid approach (Single Pass).
    It generates a fixed grid of Invariants, checks physics stability for each,
    and saves the surviving samples.
    """

    # ---------------------------------------------------------------------
    # 1. Configure Stages (Single vs Multi)
    # ---------------------------------------------------------------------
    if cfg.mode == "single_stage":
        stages = [{"target_cond": None, "stage_range": (1.0, 2.4)}]
        folder_mode = "single_stage"
    elif cfg.mode == "multi_stage":
        stages = [
            {"target_cond": None, "stage_range": (1.0, 1.8)},
            {"target_cond": None, "stage_range": (1.8, 2.0)},
            {"target_cond": None, "stage_range": (2.0, 2.2)},
            {"target_cond": None, "stage_range": (2.2, 2.4)},
        ]
        folder_mode = "multi_stage"
    else:
        raise ValueError(f"Unknown cfg.mode: {cfg.mode}")

    # ---------------------------------------------------------------------
    # 2. Main Loop Over Stages
    # ---------------------------------------------------------------------
    for stage in stages:
        stage_start = time.time()
        
        cfg.target_cond = stage["target_cond"]
        cfg.stage_range = stage["stage_range"]
        
        # Determine tag for folder naming
        if cfg.stage_range:
            stage_tag = f"{cfg.stage_range[0]}_{cfg.stage_range[1]}"
        else:
            stage_tag = f"{cfg.target_cond}"

        print(f"=== Generating Grid Uniform Maxwell-B | Mode: {cfg.mode} | Stage: {stage_tag} ===")
        
        # Reset seed per stage for consistency
        set_all_seeds(cfg.seed)

        # Storage lists
        L0_list, Y_list, condA_list, residual_list = [], [], [], []

        # -----------------------------------------------------------------
        # 3. Grid Generation Strategy (Single Pass)
        # -----------------------------------------------------------------
        # Factor 15.0: High density grid to maximize chance of hitting strict targets
        candidate_grid = generate_invariant_grid(cfg.n_samples, max_neg_II=1.5, oversample_factor=10.0)
        
        # -----------------------------------------------------------------
        # 4. Iteration Loop (Consuming the Grid)
        # -----------------------------------------------------------------
        for nII, iii in candidate_grid:
            
            # Stop if we hit target (optimization)
            if len(L0_list) >= cfg.n_samples:
                break

            # A. Recover Eigenvalues & Construct Tensor D
            evals = get_eigenvalues_from_invariants(nII, iii)
            Lambda = np.diag(evals)
            
            # Rotate to random orientation
            R = generate_random_rotation_matrix(cfg.dim)
            D = R @ Lambda @ R.T
            
            # B. Generate Random Vorticity (W)
            w_vec = np.random.randn(3)
            W = np.zeros((3, 3))
            W[0, 1], W[0, 2], W[1, 2] = -w_vec[2], w_vec[1], -w_vec[0]
            W = W - W.T
            
            # Scale W (Random ratio 0 to 1.0)
            norm_D = np.linalg.norm(D)
            norm_W = np.linalg.norm(W)
            v_ratio = np.random.uniform(0, 1.0) 
            
            if norm_W > 1e-10:
                W_scaled = W * (v_ratio * norm_D / norm_W)
            else:
                W_scaled = W
                
            L = D + W_scaled
            
            # -------------------------------------------------------------
            # 5. Maxwell-B Physics & Stability Check
            # -------------------------------------------------------------
            A = np.eye(cfg.dim) - cfg.lam * L
            cA = cond(A)
            
            accept = False
            
            # Filter: Range Mode
            if cfg.stage_range is not None:
                if cfg.stage_range[0] <= cA <= cfg.stage_range[1]:
                    accept = True
            # Filter: Target Condition Mode
            elif cfg.target_cond is not None:
                if abs(cA - cfg.target_cond) <= 0.05:
                    accept = True
            
            # -------------------------------------------------------------
            # 6. Solve & Save (UPDATED STRICT CHECK)
            # -------------------------------------------------------------
            if accept:
                B_mat = -cfg.lam * L.T
                C_mat = 2.0 * cfg.eta0 * D
                try:
                    T = solve_sylvester(A, B_mat, C_mat)
                    
                    # Residual Check (Solver Accuracy)
                    Res = A @ T + T @ B_mat - C_mat
                    resid = np.linalg.norm(Res, 'fro')
                    
                    # Symmetry Check (Physics Consistency)
                    sym_err = np.linalg.norm(T - T.T, 'fro')
                    
                    # STRICT CHECK:
                    # 1. Residual must be near machine precision (< 1e-12)
                    # 2. Symmetry error must be near machine precision (< 1e-12)
                    # This guarantees the sample is a TRUE solution to the physics.
                    if resid < 1e-12 and sym_err < 1e-12: 
                        
                        # Optional: Force exact symmetry to remove 1e-16 noise
                        T = 0.5 * (T + T.T)
                        
                        L0_list.append(L)
                        Y_list.append(T)
                        condA_list.append(cA)
                        residual_list.append(resid)
                        
                except Exception:
                    pass # Ignore solver failures

        # -----------------------------------------------------------------
        # 7. Final Double-Check & Save (CRASH FIX APPLIED)
        # -----------------------------------------------------------------
        # Only proceed if we actually found samples
        if len(L0_list) > 0:
            
            # Safe Filter
            try:
                filtered_L0_list, kept_mask = filter_admissible_region(L0_list)
                Y_list = [T for T, keep in zip(Y_list, kept_mask) if keep]
                condA_list = [c for c, keep in zip(condA_list, kept_mask) if keep]
                residual_list = [r for r, keep in zip(residual_list, kept_mask) if keep]
                L0_list = filtered_L0_list
            except Exception as e:
                print(f"⚠️ Warning: Filtering failed for stage {stage_tag}. Error: {e}")
                L0_list = []

            # Define Paths
            # --- ADD THIS LOGIC ---
            if cfg.n_samples >= 1000:
               size_folder = f"{int(cfg.n_samples/1000)}ksamples"
            else:
               size_folder = f"{cfg.n_samples}samples"
# ----------------------

            # --- MODIFY THESE PATHS ---
            # Add 'size_folder' to the end of the path
            stage_data_path = os.path.join(cfg.paths.data, "uniform", folder_mode, f"seed_{cfg.seed}", stage_tag, size_folder)
            stage_images_path = os.path.join(cfg.paths.images, "uniform", folder_mode, f"seed_{cfg.seed}", stage_tag, size_folder, "maxwell_B")
            os.makedirs(stage_data_path, exist_ok=True)
            os.makedirs(stage_images_path, exist_ok=True)

            # Convert to Numpy/Torch
            X_np = np.array(L0_list)
            Y_np = np.array(Y_list)
            
            if len(X_np) > 0:
                X_flat = X_np.reshape(X_np.shape[0], -1)
                Y_flat = flatten_symmetric_tensors(Y_np)

                # Save .pt files
                torch.save(torch.tensor(X_flat, dtype=torch.float32), os.path.join(stage_data_path, f"X_{cfg.dim}D_maxwell_B_stage.pt"))
                torch.save(torch.tensor(Y_flat, dtype=torch.float32), os.path.join(stage_data_path, f"Y_{cfg.dim}D_maxwell_B_stage.pt"))
                
                print(f"📊 Samples collected: {len(L0_list)} (from Single Grid Pass)")
                print(f"📊 X tensor shape: {torch.tensor(X_flat).shape}")
                print(f"📊 Y tensor shape: {torch.tensor(Y_flat).shape}")

                # -----------------------------------------------------------------
                # 8. Generate Analysis (Plots & Stats)
                # -----------------------------------------------------------------
                write_uniform_analysis_file(
                    model_name="maxwell_B_uniform",
                    stability_status=cfg.mode,
                    L_list=L0_list, T_list=Y_list,
                    condA_list=condA_list, lam=cfg.lam,
                    save_root=stage_images_path,
                    cfg=cfg, X_flat=X_flat, Y_flat=Y_flat,
                    stage_tag=stage_tag,
                    residual_list=residual_list
                )
            else:
                # Case: Samples existed but were all filtered out by admissible region check
                print(f"⚠️ WARNING: All samples were filtered out for stage {stage_tag}. Skipping save.")
                # We still print shape 0 to satisfy your requested print format
                print(f"📊 X tensor shape: torch.Size([0])")
                print(f"📊 Y tensor shape: torch.Size([0])")

        else:
            # Case: Grid missed target completely (Normal for Stage 1.0)
            print(f"⚠️ WARNING: No valid samples found for stage {stage_tag} (Grid missed target). Skipping save.")
            print(f"📊 X tensor shape: torch.Size([0])")
            print(f"📊 Y tensor shape: torch.Size([0])")
        
        # ---------------------------------------------------------------------
        # Timer & device info
        # ---------------------------------------------------------------------
        elapsed_time = time.time() - stage_start
        gpus = GPUtil.getGPUs()
        if gpus:
            print(f"⏱ Stage {stage_tag} time: {elapsed_time:.2f}s on {gpus[0].name}")
        else:
            print(f"⏱ Stage {stage_tag} time: {elapsed_time:.2f}s on CPU")

    # ---------------------------------------------------------------------
    # 9. Multi-Stage Progression Plot (Optional)
    # ---------------------------------------------------------------------
    if cfg.mode == "multi_stage":
        stage_tags = ["1.0_1.8", "1.8_2.0", "2.0_2.2", "2.2_2.4"]
        plot_stage_progression(
            data_root=os.path.join(cfg.paths.data, "uniform", folder_mode, f"seed_{cfg.seed}"),
            images_root=os.path.join(cfg.paths.images, "uniform", folder_mode, f"seed_{cfg.seed}"),
            model_name="maxwell_B",
            suffix="_stage",
            stages=stage_tags
        )

if __name__ == "__main__":
    main()