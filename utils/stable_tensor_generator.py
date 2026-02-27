#=================================================================================
# 1.Imports 
#=================================================================================

import numpy as np
import torch
import os
from scipy.linalg import solve_sylvester
from numpy.linalg import norm, cond
import hydra
from omegaconf import DictConfig
from utils.tensors import generate_base_L_tensor, flatten_symmetric_tensors

#===========================================================================================
# 2. Steady State Differential Models
#===========================================================================================
#********************************************************************************************************
# Maxwell -B Model
#*******************************************************************************************************

def generate_stable_sample_maxwell(dim, eta0, lam, target_cond=None, stage_range=None, tol=0.05):
    """
    Generate one stable Maxwell-B sample.
    
    Modes:
    ------
    1) Single target_cond mode:
        target_cond is a float, stage_range is None
        → matches |cond(A) - target_cond| <= tol
    
    2) Range mode:
        stage_range is (min_cond, max_cond)
        target_cond is None
        → matches cond_min <= cond(A) <= cond_max
    
    Parameters:
    -----------
    dim : int
        Dimension (usually 3 for 3D)
    eta0 : float
        Zero-shear viscosity
    lam : float
        Relaxation time
    target_cond : float or None
        Single conditioning target (Stage 1)
    stage_range : tuple(float, float) or None
        Accepted range (cond_min, cond_max) (Stage 2+)
    tol : float
        Allowed tolerance for single target_cond mode
    """

    while True:
        L0 = generate_base_L_tensor(dim=dim, vorticity_ratio=np.random.uniform(0, 1.0))
        D = 0.5 * (L0 + L0.T)
        W = 0.5 * (L0 - L0.T)
        A = np.eye(dim) - lam * L0
        cA = cond(A)

        # Mode 1: fixed target_cond
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

        # Mode 2: range
        elif stage_range is not None and target_cond is None:
            cond_min, cond_max = stage_range
            if cond_min <= cA <= cond_max and np.isfinite(cA):
                break

        else:
            raise ValueError("Either target_cond OR stage_range must be set, but not both.")

    B = -lam * L0.T
    C = 2.0 * eta0 * D
    T = solve_sylvester(A, B, C)

    return L0, D, W, T, cA
#********************************************************************************************************
# Oldroyd-B Model
#********************************************************************************************************
def generate_stable_sample_oldroyd(dim, eta0, lam, lam_r, target_cond=None, stage_range=None, tol=0.05):
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
    return L0, D, W, T, cA

#********************************************************************************************************
# Phan-Thien Tanner Exponential Model
#*******************************************************************************************************
def generate_stable_sample_ptt_exponential(dim, eta0, lam, alpha=1.0,
                                           target_cond=None, stage_range=None,
                                           max_iter=50, tol=1e-12, damping=1.0):
    """
    Generate one stable Phan-Thien–Tanner (PTT) exponential form sample.

    Supports:
    ---------
    - Single target_cond for stage 1
    - stage_range (min_cond, max_cond) for later stages

    Parameters:
    -----------
    dim : int
        Dimension (usually 3 for 3D).
    eta0 : float
        Zero-shear viscosity.
    lam : float
        Relaxation time.
    alpha : float
        Nonlinearity parameter in exponential term.
    target_cond : float or None
        Target A = I - lam * L0 condition number (single target).
    stage_range : tuple(float,float) or None
        Allowed range for condition number (for stage-range mode).
    max_iter : int
        Maximum fixed-point iterations for Step 3.
    tol : float
        Convergence tolerance for Step 3.
    damping : float
        Under-relaxation factor for stability in Step 3.
    """

    while True:
        # === Step 1: Generate random L0 with conditioning control like Maxwell_B ===
        L0 = generate_base_L_tensor(dim=dim, vorticity_ratio=np.random.uniform(0, 1.0))
        D = 0.5 * (L0 + L0.T)
        W = 0.5 * (L0 - L0.T)

        A = np.eye(dim) - lam * L0
        cA = cond(A)

        if target_cond is not None and stage_range is None:
            # Adjust L0 to meet target condition number within tolerance
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

    # === Step 2: Initial guess using Maxwell-B (ψ = 0 case) ===
    B = -lam * L0.T
    C = 2.0 * eta0 * D
    T = solve_sylvester(A, B, C)
    T = 0.5 * (T + T.T)  # enforce symmetry

    # === Step 3: Fixed-point iteration for exponential PTT ===
    max_trace_limit = 50.0  # clamp to avoid np.exp overflow

    for it in range(max_iter):
        # Nonlinearity term psi = exp(alpha * tr(T)) - 1

        trace_T = np.trace(T)
        trace_T = np.clip(trace_T, -max_trace_limit, max_trace_limit)  # clamp extremes

        psi = np.exp(alpha * trace_T) - 1.0

        # Safety check: skip bad samples if psi is NaN or inf
        if not np.isfinite(psi):
            # Regenerate new sample if unstable
            return generate_stable_sample_ptt_exponential(dim, eta0, lam, alpha,
                                                          target_cond, stage_range,
                                                          max_iter, tol, damping)

        # Effective operators for Sylvester equation with nonlinearity
        A_eff = (1.0 + psi) * np.eye(dim) - lam * L0
        B_eff = -lam * L0.T

        try:
            T_new = solve_sylvester(A_eff, B_eff, C)
        except Exception:
            # Sylvester failed — restart with new sample
            return generate_stable_sample_ptt_exponential(dim, eta0, lam, alpha,
                                                          target_cond, stage_range,
                                                          max_iter, tol, damping)

        T_new = 0.5 * (T_new + T_new.T)  # enforce symmetry

        # Optional damping for stability in nonlinear iteration
        T_new = T + damping * (T_new - T)

        # Convergence check: Frobenius norm of change
        if np.linalg.norm(T_new - T) < tol:
            T = T_new
            break

        T = T_new

    # If we reach here, return the generated stable sample
    return L0, D, W, T, cA
#=================================================================================================
# 3. Main file
#=================================================================================================
@hydra.main(config_path="../config/data", config_name="dataConfig", version_base=None)
def main(cfg: DictConfig):
    np.random.seed(cfg.seed)
    stages = [
        {"target_cond": 1.0, "stage_range": None},
        {"target_cond": None, "stage_range": (1.0, 1.2)},
        {"target_cond": None, "stage_range": (1.2, 1.4)},
        {"target_cond": None, "stage_range": (1.4, 1.6)},
        {"target_cond": None, "stage_range": (1.6, 1.8)},
        {"target_cond": None, "stage_range": (1.8, 2.0)},
        {"target_cond": None, "stage_range": (2.0, 2.2)},
        {"target_cond": None, "stage_range": (2.2, 2.4)},
        {"target_cond": None, "stage_range": (2.4, 2.6)},
        {"target_cond": None, "stage_range": (2.6, 2.8)}
        ]

    for stage in stages:
        target_cond = stage["target_cond"]
        stage_range = stage["stage_range"]
        stage_tag = (f"{stage_range[0]}_{stage_range[1]}" if stage_range else f"{target_cond}")
        stage_data_path = os.path.join(cfg.paths.data, stage_tag)
        os.makedirs(stage_data_path, exist_ok=True)

        L0_list, T_list = [], []
        for i in range(cfg.n_samples):
            if cfg.constitutive_eq == "maxwell_B":
                L0, D, W, T, condA = generate_stable_sample_maxwell(
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
        suffix = "_stable"
        torch.save(torch.tensor(X_flat, dtype=torch.float32),
                   os.path.join(stage_data_path, f"X_{cfg.dim}D_{cfg.constitutive_eq}{suffix}.pt"))
        torch.save(torch.tensor(Y_flat, dtype=torch.float32),
                   os.path.join(stage_data_path, f"Y_{cfg.dim}D_{cfg.constitutive_eq}{suffix}.pt"))

        print(f"✅ Stage {stage_tag} saved: X shape {X_flat.shape}, Y shape {Y_flat.shape}")

if __name__ == "__main__":
    main()