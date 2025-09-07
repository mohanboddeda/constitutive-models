import os
import numpy as np
import torch
import hydra
from omegaconf import DictConfig

# ==== Import model solvers ====
from model.maxwell import solve_steady_state_maxwell
from model.oldroyd import solve_steady_state_oldroyd
from model.carreau import carreau_yasuda_viscosity

# ==== Import utilities ====
from utils.tensors import generate_base_L_tensor, flatten_symmetric_tensors
from utils.plotting import (
    plot_tensor_matrix,
    plot_invariants_diagram,
    plot_carreau_parameter_study  # NEW import for Carreau plotting
)
from utils.residual_checks import carreau_yasuda_residual_check
# -----------------------------------------------------------------------------
# Main Loop
#------------------------------------------------------------------------------

@hydra.main(config_path="config/data", config_name="dataConfig", version_base=None)
def main(cfg: DictConfig) -> None:
    # -------------------------------------------------------------------------
    # 1) Create folders for images and data
    # -------------------------------------------------------------------------
    os.makedirs(cfg.paths.images, exist_ok=True)
    os.makedirs(cfg.paths.data, exist_ok=True)

    L0_list, Y_list = [], []
    max_cond_allowed = 1e8  # Condition number threshold

    # For Carreau color scaling
    if cfg.constitutive_eq == "carreau_yasuda": 
        nu_min, nu_max = float('inf'), float('-inf')

    # -------------------------------------------------------------------------
    # 2) Sampling loop
    # -------------------------------------------------------------------------
    while len(L0_list) < cfg.n_samples:
        v_ratio = np.random.uniform(0, cfg.max_vorticity_ratio)
        L0 = generate_base_L_tensor(dim=cfg.dim, vorticity_ratio=v_ratio)

        if cfg.constitutive_eq == "carreau_yasuda":
            nu = carreau_yasuda_viscosity(
                L0,
                nu_0=5.28e-5, nu_inf=3.30e-6,
                lambda_val=1.902, n=0.22, a=1.25
            )
            L0_list.append(L0)
            Y_list.append(nu)
            nu_min = min(nu_min, nu)
            nu_max = max(nu_max, nu)
            

        elif cfg.constitutive_eq == "maxwell_B":
           debug_flag = (len(L0_list) < 3)

           T, condM, resid = solve_steady_state_maxwell(
           L0,
           return_cond=True,
           debug=debug_flag,
           sample_idx=len(L0_list),
           use_projection=cfg.use_projection
           )

           if condM > max_cond_allowed or resid > 1e-12:
             continue

           L0_list.append(L0)
           Y_list.append(T)

        elif cfg.constitutive_eq == "oldroyd_B":
           debug_flag = (len(L0_list) < 3)

           T, condM, resid = solve_steady_state_oldroyd(
           L0,
           eta0=5.28e-5, lam=1.902, lam_r=1.0,
           return_cond=True,
           debug=debug_flag,
           sample_idx=len(L0_list),
           use_projection=cfg.use_projection
           )

           if condM > max_cond_allowed or resid > 1e-12:
             continue

           L0_list.append(L0)
           Y_list.append(T)

        else:
            raise ValueError(f"Unknown constitutive_eq: {cfg.constitutive_eq}")

    # -------------------------------------------------------------------------
    # 3) Debug plotting – first 3 accepted samples
    # -------------------------------------------------------------------------
    for i in range(min(3, len(L0_list))):
        L0 = L0_list[i]
        plot_tensor_matrix(
            L0,
            title=f"L0 sample {i}",
            filename=os.path.join(cfg.paths.images, f"L0_sample_{i}.png")
        )
        if cfg.constitutive_eq == "carreau_yasuda":
            nu = Y_list[i]
            print(f"Viscosity for sample {i}: {nu:.3e} Pa·s")
        elif cfg.constitutive_eq == "maxwell_B":
            plot_tensor_matrix(
                Y_list[i],
                model_name="Maxwell-B",
                title=f"Maxwell-B Stress Tensor sample {i}",
                filename=os.path.join(cfg.paths.images, f"T_maxwell_sample_{i}.png")
            )
        elif cfg.constitutive_eq == "oldroyd_B":
            plot_tensor_matrix(
                Y_list[i],
                model_name="Oldroyd-B",
                title=f"Oldroyd-B Stress Tensor sample {i}",
                filename=os.path.join(cfg.paths.images, f"T_oldroyd_sample_{i}.png")
            )
    # Run Carreau–Yasuda residual validation for 3 random samples
    if cfg.constitutive_eq == "carreau_yasuda": 
        carreau_yasuda_residual_check(L0_list, Y_list)        

    # -------------------------------------------------------------------------
    # 4) Flatten tensors for saving
    # -------------------------------------------------------------------------
    X_np = np.array(L0_list)
    Y_np = np.array(Y_list)

    if cfg.constitutive_eq == "carreau_yasuda":
        X_flat = X_np.reshape(X_np.shape[0], -1)
        Y_flat = Y_np.reshape(-1, 1)
    elif cfg.constitutive_eq in ["maxwell_B", "oldroyd_B"]:
        X_flat = X_np.reshape(X_np.shape[0], -1)
        Y_flat = flatten_symmetric_tensors(Y_np)

    # -------------------------------------------------------------------------
    # 5) Invariants diagram
    # -------------------------------------------------------------------------
    plot_invariants_diagram(
        cfg,
        L0_list,
        title=f"{cfg.dim}D Strömungsarten im Invariantenraum",
        paths=True
    )

    # -------------------------------------------------------------------------
    # 6) Save datasets
    # -------------------------------------------------------------------------
    data_path_X = os.path.join(cfg.paths.data, f"X_{cfg.dim}D_{cfg.constitutive_eq}.pt")
    data_path_Y = os.path.join(cfg.paths.data, f"Y_{cfg.dim}D_{cfg.constitutive_eq}.pt")
    torch.save(torch.tensor(X_flat, dtype=torch.float32), data_path_X)
    torch.save(torch.tensor(Y_flat, dtype=torch.float32), data_path_Y)

    # -------------------------------------------------------------------------
    # 7) Carreau-Yasuda parameter study plot
    # -------------------------------------------------------------------------
    if cfg.constitutive_eq == "carreau_yasuda":
        gamma_list = []
        epsilon = 1e-12
        for L0 in L0_list:
            D = 0.5 * (L0 + L0.T)
            from utils.invariants import compute_invariants_vectorized
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
            (0.25, 100.0, 0.5),
        ]
        save_path = os.path.join(cfg.paths.images, "carreau_parameter_study.png")
        plot_carreau_parameter_study(
            gamma_list,
            nu_vals,
            nu_0=5.28e-5,
            nu_inf=3.30e-6,
            param_sets=param_sets,
            save_path=save_path
        )

    # -------------------------------------------------------------------------
    # 8) Console inspection
    # -------------------------------------------------------------------------
    print("\n=== First 5 X samples ===")
    print(np.array_str(X_flat[:5], precision=4, suppress_small=True))
    print("\n=== First 5 Y samples ===")
    print(np.array_str(Y_flat[:5], precision=6, suppress_small=True))
    print("\n=== X column min/max ===")
    print("Min:", np.min(X_flat, axis=0))
    print("Max:", np.max(X_flat, axis=0))
    print("\n=== Y column min/max ===")
    print("Min:", np.min(Y_flat, axis=0))
    print("Max:", np.max(Y_flat, axis=0))
    print("\nFertig")
    print(f"-> X shape: {X_flat.shape}")
    print(f"-> Y shape: {Y_flat.shape}")

if __name__ == "__main__":
    main()