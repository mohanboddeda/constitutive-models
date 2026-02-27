import os
import numpy as np
from numpy.linalg import cond
from scipy.linalg import solve_sylvester
import hydra
import torch
from omegaconf import DictConfig

# === Project utilities ===
from utils.tensors import flatten_symmetric_tensors
from utils.plotting import analyze_and_plot_dataset  # your plotting script

def compute_T_maxwell(L0, eta0, lam, target_cond):
    """
    Compute Maxwell-B steady state tensor T from L.
    Returns:
        T       (ndarray) : extra stress tensor
        D       (ndarray) : symmetric part of L
        cA      (float)   : condition number of A
        resid   (float)   : Frobenius norm of residual
        sym_err (float)   : Frobenius norm of symmetry error
    """
    # --- Scale to target condition number ---
    A = np.eye(L0.shape[0]) - lam * L0
    cA = cond(A)
    if abs(cA - target_cond) > 0.05:
        scale_factor = (cA / target_cond) ** (-1)
        L0 *= scale_factor
        A = np.eye(L0.shape[0]) - lam * L0
        cA = cond(A)

    # --- Decompose ---
    B = -lam * L0.T
    D = 0.5 * (L0 + L0.T)
    C = 2.0 * eta0 * D

    # --- Solve Sylvester equation ---
    T = solve_sylvester(A, B, C)

    # --- Residual and symmetry error ---
    R = A @ T + T @ B - C
    resid = np.linalg.norm(R, 'fro')
    sym_err = np.linalg.norm(T - T.T, 'fro')

    return T, D, cA, resid, sym_err

@hydra.main(config_path="config/data", config_name="maxiconfig", version_base=None)
def main(cfg: DictConfig):
    np.random.seed(cfg.seed)

    # Get flow types from config or detect automatically
    flow_types = cfg.get("flow_types", None)
    if flow_types is None:
        flow_types = [d for d in os.listdir("flow_data") if os.path.isdir(os.path.join("flow_data", d))]

    for flow_type in flow_types:
        print(f"\n=== Processing flow type: {flow_type} ===")
        folder_path = os.path.join("flow_data", flow_type)
        L_path = os.path.join(folder_path, "L.npy")

        if not os.path.exists(L_path):
            print(f"❌ Missing L.npy in {folder_path}, skipping...")
            continue

        # === Load L dataset ===
        L_array = np.load(L_path)  # shape: (n_samples, dim, dim)
        T_list = []

        # === Create stats .txt file ===
        stats_txt_path = os.path.join(folder_path, f"{flow_type}_T_stats.txt")
        with open(stats_txt_path, "w") as ftxt:
            ftxt.write(f"=== Maxwell-B T computation for {flow_type} ===\n")
            ftxt.write(f"eta0 = {cfg.eta0}, lambda = {cfg.lambda_}, target_cond = {cfg.target_cond}\n")
            ftxt.write(f"Number of samples: {L_array.shape[0]}\n\n")

            for i, L_matrix in enumerate(L_array):
                T_matrix, D_matrix, condA, resid, sym_err = compute_T_maxwell(
                    L_matrix.copy(), cfg.eta0, cfg.lambda_, cfg.target_cond
                )
                T_list.append(T_matrix)

                # ✅ Only first N samples printed in .txt
                if i < cfg.debug_print_samples:
                    ftxt.write("="*40 + "\n")
                    ftxt.write(f"Sample {i}:\n")
                    ftxt.write(f"Condition number(A) = {condA:.4f}\n")
                    ftxt.write("L matrix:\n" + np.array_str(L_matrix, precision=4, suppress_small=True) + "\n")
                    ftxt.write(f"L norm = {np.linalg.norm(L_matrix, 'fro'):.4e}\n")
                    ftxt.write("D matrix:\n" + np.array_str(D_matrix, precision=4, suppress_small=True) + "\n")
                    ftxt.write(f"D norm = {np.linalg.norm(D_matrix, 'fro'):.4e}\n")
                    ftxt.write("T matrix:\n" + np.array_str(T_matrix, precision=4, suppress_small=True) + "\n")
                    ftxt.write(f"Residual Frobenius norm = {resid:.4e}\n")
                    ftxt.write(f"T symmetry error Frobenius norm = {sym_err:.4e}\n")
                    ftxt.write(f"T norm = {np.linalg.norm(T_matrix, 'fro'):.4e}\n")
                    
        print(f"📄 Saved stats to {stats_txt_path}")

        # === Save T and flatten Y in .pt format ===
        T_array = np.array(T_list)
        Y_flat = flatten_symmetric_tensors(T_array)
        X_flat = L_array.reshape(L_array.shape[0], -1)

        suffix = "_stable"  # or "_unstable" depending on cfg.stable
        torch.save(torch.tensor(X_flat, dtype=torch.float32),
               os.path.join(folder_path, f"X_{cfg.dim}D_maxwell_B{suffix}.pt"))

        torch.save(torch.tensor(Y_flat, dtype=torch.float32),
               os.path.join(folder_path, f"Y_{cfg.dim}D_maxwell_B{suffix}.pt"))

        print(f"💾 Saved L0 as X_{cfg.dim}D_maxwell_B{suffix}.pt")
        print(f"💾 Saved T as Y_{cfg.dim}D_maxwell_B{suffix}.pt")

        # === Create per-flow-type plots folder ===
        plots_dir = os.path.join(folder_path, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Override cfg.paths.images so plots go into the per-flow-type folder
        cfg.paths.images = plots_dir

        # === Run analysis and plotting ===
        analyze_and_plot_dataset(X_flat, Y_flat, cfg)

if __name__ == "__main__":
    main()