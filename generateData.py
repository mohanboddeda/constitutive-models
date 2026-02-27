import os
import numpy as np
import torch
import hydra
import time
import GPUtil
from omegaconf import DictConfig
import random

# ==== Fix seed ====
def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==== Import model solvers ====
from model.maxwell import solve_steady_state_maxwell
from model.oldroyd import solve_steady_state_oldroyd
from model.carreau import carreau_yasuda_viscosity

# ==== Import utilities ====
from utils.tensors import generate_base_L_tensor, flatten_symmetric_tensors
from utils.write_sampledata import write_sampledata_file 
from utils.invariants import filter_admissible_region, compute_invariants_vectorized
from utils.plot_stage_progression import plot_stage_progression
from utils.stable_tensor_generator import (
    generate_stable_sample_maxwell,
    generate_stable_sample_oldroyd,
    generate_stable_sample_ptt_exponential
)
from utils.plotting import plot_carreau_parameter_study

# -------------------------------------------------------------------------
# Main Hydra entry point
# -------------------------------------------------------------------------

@hydra.main(config_path="config/data", config_name="dataConfig", version_base=None)
def main(cfg: DictConfig) -> None:
    # -------------------------------------------------------------------------
    # Stage definitions – only used for stable case
    # -------------------------------------------------------------------------
    if cfg.stable:
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
    else:
        # Unstable: just one "stage" placeholder
        stages = [{"target_cond": None, "stage_range": None}]

    # -------------------------------------------------------------------------
    # Loop over stages (or single iteration if unstable)
    # -------------------------------------------------------------------------
    for stage in stages:
        stage_start = time.time()
        print(f"=== Generating {cfg.constitutive_eq.replace('_', '-').title()}_{'Stable' if cfg.stable else 'Unstable'} Stage: target_cond={stage['target_cond']} range={stage['stage_range']} ===")
         
        cfg.target_cond = stage["target_cond"]
        cfg.stage_range = stage["stage_range"]

        # Seed for reproducibility
        set_all_seeds(cfg.get("seed", 42))

        L0_list, Y_list, condA_list, W_list = [], [], [], []
        max_cond_allowed = 1e8

        if cfg.constitutive_eq == "carreau_yasuda":
            nu_min, nu_max = float('inf'), float('-inf')

        # -------------------- Sampling loop --------------------
        while len(L0_list) < cfg.n_samples:
            if not cfg.stable:
                # Randomly generate L0 (unstable case)
                v_ratio = np.random.uniform(0, cfg.max_vorticity_ratio)
                L0 = generate_base_L_tensor(dim=cfg.dim, vorticity_ratio=v_ratio)

            if cfg.constitutive_eq == "carreau_yasuda":
                # Carreau-Yasuda viscosity model
                nu = carreau_yasuda_viscosity(
                    L0,
                    nu_0=5.28e-5, nu_inf=3.30e-6,
                    lambda_val=1.902, n=0.22, a=1.25
                )
                L0_list.append(L0)
                Y_list.append(nu)
                condA_list.append(None)
                nu_min = min(nu_min, nu)
                nu_max = max(nu_max, nu)

            elif cfg.constitutive_eq == "maxwell_B":
                if cfg.stable:
                    L0, D, W, T, condA = generate_stable_sample_maxwell(
                        cfg.dim, cfg.eta0, cfg.lam,
                        target_cond=cfg.target_cond,
                        stage_range=cfg.stage_range
                    )
                    L0_list.append(L0); Y_list.append(T); condA_list.append(condA)
                else:
                    T, condM, resid = solve_steady_state_maxwell(
                        L0,
                        eta0=cfg.eta0,
                        lam=cfg.lam,
                        return_cond=True,
                        debug=(len(L0_list) < 3),
                        sample_idx=len(L0_list),
                        use_projection=cfg.use_projection
                    )
                    if condM > max_cond_allowed or resid > 1e-12:
                        continue
                    D = 0.5 * (L0 + L0.T)
                    W = 0.5 * (L0 - L0.T)
                    L0_list.append(L0); Y_list.append(T)
                    condA_list.append(condM); W_list.append(W)

            elif cfg.constitutive_eq == "oldroyd_B":
                if cfg.stable:
                    L0, D, W, T, condA = generate_stable_sample_oldroyd(
                        cfg.dim, cfg.eta0, cfg.lam, cfg.lam_r,
                        target_cond=cfg.target_cond,
                        stage_range=cfg.stage_range
                    )
                    L0_list.append(L0); Y_list.append(T); condA_list.append(condA)
                else:
                    T, condM, resid = solve_steady_state_oldroyd(
                        L0,
                        eta0=cfg.eta0,
                        lam=cfg.lam,
                        lam_r=cfg.lam_r,
                        return_cond=True,
                        debug=(len(L0_list) < 3),
                        sample_idx=len(L0_list),
                        use_projection=cfg.use_projection
                    )
                    if condM > max_cond_allowed or resid > 1e-12:
                        continue
                    D = 0.5 * (L0 + L0.T)
                    W = 0.5 * (L0 - L0.T)
                    L0_list.append(L0); Y_list.append(T)
                    condA_list.append(condM); W_list.append(W)

            elif cfg.constitutive_eq == "ptt_exponential":
                if cfg.stable:
                    L0, D, W, T, condA = generate_stable_sample_ptt_exponential(
                        cfg.dim, cfg.eta0, cfg.lam, alpha=1.0,
                        target_cond=cfg.target_cond,
                        stage_range=cfg.stage_range
                    )
                    L0_list.append(L0); Y_list.append(T); condA_list.append(condA)
                else:
                    raise NotImplementedError("Unstable PTT exponential not implemented.")

        # -------------------- Filtering --------------------
        filtered_L0_list, kept_mask = filter_admissible_region(L0_list)
        Y_list = [T for T, keep in zip(Y_list, kept_mask) if keep]
        condA_list = [cond for cond, keep in zip(condA_list, kept_mask) if keep]
        W_list = [W for W, keep in zip(W_list, kept_mask) if keep]
        L0_list = filtered_L0_list

        # -------------------- Path setup --------------------
        if cfg.stable:
            if cfg.stage_range:
                stage_tag = f"{cfg.stage_range[0]}_{cfg.stage_range[1]}"
            else:
                stage_tag = f"{cfg.target_cond}"
            stage_data_path = os.path.join(cfg.paths.data, stage_tag)
            stage_images_path = os.path.join(cfg.paths.images, f"{stage_tag} stable", cfg.constitutive_eq)
        else:
            stage_tag = "unstable"
            stage_data_path = os.path.join(cfg.paths.data, stage_tag)
            stage_images_path = os.path.join(cfg.paths.images, stage_tag, cfg.constitutive_eq)

        os.makedirs(stage_data_path, exist_ok=True)
        os.makedirs(stage_images_path, exist_ok=True)

        # -------------------- Save data --------------------
        X_np = np.array(L0_list)
        Y_np = np.array(Y_list)
        if cfg.constitutive_eq == "carreau_yasuda":
            X_flat = X_np.reshape(X_np.shape[0], -1)
            Y_flat = Y_np.reshape(-1, 1)
        else:
            X_flat = X_np.reshape(X_np.shape[0], -1)
            Y_flat = flatten_symmetric_tensors(Y_np)

        suffix = "_stable" if cfg.stable else "_unstable"
        torch.save(torch.tensor(X_flat, dtype=torch.float32),
                   os.path.join(stage_data_path, f"X_{cfg.dim}D_{cfg.constitutive_eq}{suffix}.pt"))
        torch.save(torch.tensor(Y_flat, dtype=torch.float32),
                   os.path.join(stage_data_path, f"Y_{cfg.dim}D_{cfg.constitutive_eq}{suffix}.pt"))
        print(f"📊 Samples accepted: {len(L0_list)}")
        
        # -------------------- Diagnostics & Plots --------------------
        write_sampledata_file(
            model_name=cfg.constitutive_eq,
            stability_status="Stable" if cfg.stable else "Unstable",
            L_list=L0_list, T_list=Y_list,
            condA_list=condA_list, lam=cfg.lam,
            save_root=stage_images_path,
            cfg=cfg, X_flat=X_flat, Y_flat=Y_flat,
            stage_tag=stage_tag
        )
        # Tensor shape info
        print(f"📊 X tensor shape: {torch.tensor(X_flat).shape}")
        print(f"📊 Y tensor shape: {torch.tensor(Y_flat).shape}")
        
        # ===== END per-stage timer and device info =====
        elapsed_time = time.time() - stage_start
        gpus = GPUtil.getGPUs()
        if gpus:
           gpu_name = gpus[0].name
           print(f"⏱ Stage {stage_tag} generation time: {elapsed_time:.2f}s on {gpu_name}")
        else:
           print(f"⏱ Stage {stage_tag} generation time: {elapsed_time:.2f}s on CPU")

        if cfg.constitutive_eq == "carreau_yasuda":
            gamma_list = []
            epsilon = 1e-12
            for L0 in L0_list:
                D = 0.5 * (L0 + L0.T)
                _, second_invariant_D, _ = compute_invariants_vectorized(D)
                second_invariant_D = -second_invariant_D
                shear_rate = 2 * np.sqrt(second_invariant_D + epsilon)[0]
                gamma_list.append(shear_rate)
            nu_vals = np.array(Y_list).flatten()
            param_sets = [
                (0.25, 1.0, 2.0),
                (0.5, 1.0, 2.0),
                (0.25, 100.0, 2.0),
                (0.5, 100.0, 2.0),
                (0.25, 100.0, 0.5)
            ]
            save_path = os.path.join(stage_images_path, "carreau_parameter_study.png")
            plot_carreau_parameter_study(gamma_list, nu_vals,
                                         nu_0=5.28e-5, nu_inf=3.30e-6,
                                         param_sets=param_sets, save_path=save_path)

    # ---------------------------------------------------------------------
    # Stage progression plots only for stable mode
    # ---------------------------------------------------------------------
    if cfg.stable:
        stage_tags = [
            "1.0", "1.0_1.2", "1.2_1.4", "1.4_1.6", "1.6_1.8",
            "1.8_2.0", "2.0_2.2", "2.2_2.4", "2.4_2.6",
            "2.6_2.8"
        ]
        plot_stage_progression(
            data_root=cfg.paths.data,
            images_root=cfg.paths.images,
            model_name=cfg.constitutive_eq,
            suffix="_stable",
            stages=stage_tags
        )
    

if __name__ == "__main__":
    main()