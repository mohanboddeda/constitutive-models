#=================================================================================
# 1. Imports 
#=================================================================================

import numpy as np
import torch
import os
from scipy.linalg import solve_sylvester
from numpy.linalg import cond
import hydra
from omegaconf import DictConfig
from utils.tensors import generate_base_L_tensor, flatten_symmetric_tensors

#===========================================================================================
# 2. Steady State Differential Models
#===========================================================================================

def generate_stable_sample_maxwell(dim, eta0, lam, target_cond=None, stage_range=None, tol=0.05):
    """
    Generate one stable Maxwell-B sample.
    
    CRITICAL UPDATE:
    The solver and residual checks are now INSIDE the loop. 
    This prevents returning samples that satisfy the condition number but fail the physics solver.
    """
    while True:
        # 1. Generate Candidate L
        L0 = generate_base_L_tensor(dim=dim, vorticity_ratio=np.random.uniform(0, 1.0))
        D = 0.5 * (L0 + L0.T)
        W = 0.5 * (L0 - L0.T)
        A = np.eye(dim) - lam * L0
        cA = cond(A)

        # 2. Check Stability Criteria (Mode 1: Fixed Target)
        if target_cond is not None and stage_range is None:
            # Apply Scaling if needed
            if abs(cA - target_cond) > tol:
                scale_factor = (cA / target_cond) ** (-1)
                L0 *= scale_factor
                D = 0.5 * (L0 + L0.T)
                W = 0.5 * (L0 - L0.T)
                A = np.eye(dim) - lam * L0
                cA = cond(A)
            
            # Check if we missed the target after scaling
            if not (np.isfinite(cA) and abs(cA - target_cond) <= tol):
                continue # Retry with new L0

        # 2. Check Stability Criteria (Mode 2: Range)
        elif stage_range is not None and target_cond is None:
            cond_min, cond_max = stage_range
            # Check if we are outside range
            if not (cond_min <= cA <= cond_max and np.isfinite(cA)):
                continue # Retry with new L0

        else:
            raise ValueError("Either target_cond OR stage_range must be set, but not both.")

        # =========================================================================
        # 3. Solve Physics & Verify Accuracy (MOVED INSIDE LOOP)
        # =========================================================================
        # We only reach here if Condition Number is valid. 
        # Now we check if the Solver actually works for this matrix.
        B = -lam * L0.T
        C = 2.0 * eta0 * D
        
        try:
            T = solve_sylvester(A, B, C)
            
            # Calculate Residual: || AT + TB - C ||
            R = A @ T + T @ B - C
            resid = np.linalg.norm(R, 'fro')
            
            # Calculate Symmetry Error: || T - T.T ||
            sym_err = np.linalg.norm(T - T.T, 'fro')

            # STRICT CHECK:
            # 1. Residual must be near machine precision (< 1e-12)
            # 2. Symmetry error must be near machine precision (< 1e-12)
            if resid < 1e-12 and sym_err < 1e-12:
                # Success! The sample is stable AND physically correct.
                # Optional: Clean up tiny 1e-16 noise
                T = 0.5 * (T + T.T)
                break 
            
            # If we fail the check, the loop continues and generates a new L0.
            
        except Exception:
            # If solver crashes, ignore and try again
            continue

    return L0, D, W, T, cA, resid
    
#********************************************************************************************************
# Oldroyd-B Model
#********************************************************************************************************
def generate_stable_sample_oldroyd(dim, eta0, lam, lam_r, target_cond=None, stage_range=None, tol=0.05):
    """
    Generate one stable Oldroyd-B sample. Same mode logic as Maxwell-B.
    """
    while True:
        L0 = generate_base_L_tensor(dim=dim, vorticity_ratio=np.random.uniform(0, 1.0))
        D = 0.5 * (L0 + L0.T)
        W = 0.5 * (L0 - L0.T)
        A = np.eye(dim) - lam * L0
        cA = cond(A)

        if target_cond is not None and stage_range is None:
            if abs(cA - target_cond) > tol:
                scale_factor = (cA / target_cond) ** (-1)
                L0 *= scale_factor
                D = 0.5 * (L0 + L0.T)
                W = 0.5 * (L0 - L0.T)
                A = np.eye(dim) - lam * L0
                cA = cond(A)
            if np.isfinite(cA) and abs(cA - target_cond) <= tol:
                break

        elif stage_range is not None and target_cond is None:
            cond_min, cond_max = stage_range
            if cond_min <= cA <= cond_max and np.isfinite(cA):
                break

        else:
            raise ValueError("Either target_cond OR stage_range must be set, but not both.")

    B = -lam * L0.T
    C = 2 * eta0 * (D - lam_r * (L0 @ D) - lam_r * (D @ L0.T))
    T = solve_sylvester(A, B, C)

    # Debug: Residual & Symmetry
    #R = A @ T + T @ B - C
    #resid = np.linalg.norm(R, 'fro')
    #sym_err = np.linalg.norm(T - T.T, 'fro')
   
    return L0, D, W, T, cA

#********************************************************************************************************
# Phan-Thien Tanner Exponential Model
#*******************************************************************************************************
def generate_stable_sample_ptt_exponential(dim, eta0, lam, alpha=3,
                                           target_cond=None, stage_range=None,
                                           max_iter=50, tol=1e-12, damping=1.0):
    """
    Generate one stable Phan-Thien–Tanner (PTT) exponential form sample.
    Same mode logic as above, but with nonlinear fixed-point iteration in Step 3.
    """
    while True:
        # Step 1: initial L0 generation
        L0 = generate_base_L_tensor(dim=dim, vorticity_ratio=np.random.uniform(0, 1.0))
        D = 0.5 * (L0 + L0.T)
        W = 0.5 * (L0 - L0.T)
        A = np.eye(dim) - lam * L0
        cA = cond(A)

        if target_cond is not None and stage_range is None:
            if abs(cA - target_cond) > 0.05:
                scale_factor = (cA / target_cond) ** (-1)
                L0 *= scale_factor
                D = 0.5 * (L0 + L0.T)
                W = 0.5 * (L0 - L0.T)
                A = np.eye(dim) - lam * L0
                cA = cond(A)
            if np.isfinite(cA) and abs(cA - target_cond) <= 0.05:
                break
        elif stage_range is not None and target_cond is None:
            cond_min, cond_max = stage_range
            if cond_min <= cA <= cond_max and np.isfinite(cA):
                break
        else:
            raise ValueError("Either target_cond OR stage_range must be set, but not both.")

    # Step 2: initial T guess from Maxwell-B
    B = -lam * L0.T
    C = 2.0 * eta0 * D
    T = solve_sylvester(A, B, C)
    
    # Step 3: nonlinear iteration
    max_trace_limit = 20.0
    for _ in range(max_iter):
        trace_T = np.clip(np.trace(T), -max_trace_limit, max_trace_limit)
        psi = np.exp(alpha * trace_T) - 1.0
        if not np.isfinite(psi):
            return generate_stable_sample_ptt_exponential(
                dim, eta0, lam, alpha, target_cond, stage_range,
                max_iter, tol, damping
            )

        A_eff = (1.0 + psi) * np.eye(dim) - lam * L0
        B_eff = -lam * L0.T
        try:
            T_new = solve_sylvester(A_eff, B_eff, C)
        except Exception:
            return generate_stable_sample_ptt_exponential(
                dim, eta0, lam, alpha, target_cond, stage_range,
                max_iter, tol, damping
            )
        #T_new = 0.5 * (T_new + T_new.T)
        T_new = T + damping * (T_new - T)

        if np.linalg.norm(T_new - T) < tol:
            T = T_new
            break
        T = T_new

    # === Residuals and Symmetry Check ===
    #R = A_eff @ T_new + T_new @ B_eff - C
    #resid = np.linalg.norm(R, 'fro')
    #sym_err = np.linalg.norm(T_new - T_new.T, 'fro')
    

    return L0, D, W, T, cA

#=================================================================================================
# 3. Main file
#=================================================================================================
@hydra.main(config_path="../config/data", config_name="dataConfig1", version_base=None)
def main(cfg: DictConfig):
    """
    Hydra main entry point.
    This now supports two modes from config: 'single_stage' and 'multi_stage'.
    It also saves data into seed-specific folders to avoid overwriting.
    """
    np.random.seed(cfg.seed)

    # Choose stages based on mode
    if cfg.mode == "single_stage":
        stages = [{"target_cond": None, "stage_range": (1.0, 2.4)}]
        folder_mode = "single_stage"
    elif cfg.mode == "multi_stage":
        stages = [
            {"target_cond": 1.0, "stage_range": None},
            {"target_cond": None, "stage_range": (1.0, 1.2)},
            {"target_cond": None, "stage_range": (1.2, 1.4)},
            {"target_cond": None, "stage_range": (1.4, 1.6)},
            {"target_cond": None, "stage_range": (1.6, 1.8)},
            {"target_cond": None, "stage_range": (1.8, 2.0)},
            {"target_cond": None, "stage_range": (2.0, 2.2)},
            {"target_cond": None, "stage_range": (2.2, 2.4)},
            ]
        folder_mode = "multi_stage"
    else:
        raise ValueError("cfg.mode must be 'single_stage' or 'multi_stage'")

    # Loop over each stage
    for stage in stages:
        target_cond = stage["target_cond"]
        stage_range = stage["stage_range"]
        stage_tag = f"{stage_range[0]}_{stage_range[1]}" if stage_range else f"{target_cond}"

        # Construct save path (mode + seed)
        stage_data_path = os.path.join(
            cfg.paths.data,
            folder_mode,
            f"seed_{cfg.seed}",
            stage_tag
        )
        os.makedirs(stage_data_path, exist_ok=True)

        L0_list, T_list = [], []
        

        # Sampling loop
        for _ in range(cfg.n_samples):
            if cfg.constitutive_eq == "maxwell_B":
                L0, D, W, T, condA, resid = generate_stable_sample_maxwell(
                    cfg.dim, cfg.eta0, cfg.lam,
                    target_cond=target_cond, stage_range=stage_range
                )
            elif cfg.constitutive_eq == "oldroyd_B":
                L0, D, W, T, condA = generate_stable_sample_oldroyd(
                    cfg.dim, cfg.eta0, cfg.lam, cfg.lam_r,
                    target_cond=target_cond, stage_range=stage_range
                )
            elif cfg.constitutive_eq == "ptt_exponential":
                L0, D, W, T, condA = generate_stable_sample_ptt_exponential(
                    cfg.dim, cfg.eta0, cfg.lam, alpha=1.0,
                    target_cond=target_cond, stage_range=stage_range
                )
            else:
                raise ValueError(f"Unsupported model: {cfg.constitutive_eq}")

            L0_list.append(L0)
            T_list.append(T)
            

        # Flatten and save
        X_np = np.array(L0_list)
        Y_np = np.array(T_list)
        X_flat = X_np.reshape(X_np.shape[0], -1)
        Y_flat = flatten_symmetric_tensors(Y_np)
        suffix = "_stage"
        torch.save(torch.tensor(X_flat, dtype=torch.float32),
                   os.path.join(stage_data_path, f"X_{cfg.dim}D_{cfg.constitutive_eq}{suffix}.pt"))
        torch.save(torch.tensor(Y_flat, dtype=torch.float32),
                   os.path.join(stage_data_path, f"Y_{cfg.dim}D_{cfg.constitutive_eq}{suffix}.pt"))

        print(f"✅ Stage {stage_tag} saved in {stage_data_path}: X shape {X_flat.shape}, Y shape {Y_flat.shape}")
        

if __name__ == "__main__":
    main()