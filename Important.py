def generate_stable_sample_maxwell(dim, target_cond, eta0, lam):
    """
    Generate one stable Maxwell-B sample using conditioning on A = I - lam*L0.
    """
    while True:
        L0 = generate_base_L_tensor(dim=dim, vorticity_ratio=np.random.uniform(0, 1.0))
        D = 0.5 * (L0 + L0.T)
        W = 0.5 * (L0 - L0.T)

        A = np.eye(dim) - lam * L0
        cA = cond(A)

        if abs(cA - target_cond) > 0.05:
            scale_factor = (cA / target_cond) ** (-1)
            L0 *= scale_factor
            D = 0.5 * (L0 + L0.T)
            W = 0.5 * (L0 - L0.T)
            A = np.eye(dim) - lam * L0
            cA = cond(A)
        #Repeat until cA ~ target_cond
        if np.isfinite(cA) and abs(cA - target_cond) < 0.05: 
            break

    B = -lam * L0.T
    C = 2.0 * eta0 * D
    T = solve_sylvester(A, B, C)

    return L0, D, W, T, cA

# ============================================================================
# Plotting Utilities for inverse modelling 
# ============================================================================
def vec9_to_square3(vec):
    """
    Convert a 9-component vector into a 3x3 velocity gradient tensor.
    Assumes vec = [L00, L01, L02, L10, L11, L12, L20, L21, L22]
    """
    import jax.numpy as jnp
    T = jnp.zeros((vec.shape[0], 3, 3))
    T = T.at[:, 0, 0].set(vec[:, 0])
    T = T.at[:, 0, 1].set(vec[:, 1])
    T = T.at[:, 0, 2].set(vec[:, 2])
    T = T.at[:, 1, 0].set(vec[:, 3])
    T = T.at[:, 1, 1].set(vec[:, 4])
    T = T.at[:, 1, 2].set(vec[:, 5])
    T = T.at[:, 2, 0].set(vec[:, 6])
    T = T.at[:, 2, 1].set(vec[:, 7])
    T = T.at[:, 2, 2].set(vec[:, 8])
    return T

def plot_velocitygradient_tensor_comparison(vec9_to_square3, x_true_phys, x_pred_phys, sample_indices, save_dir, model_type):
    """
    Plot comparison between true and predicted velocity gradient tensors.
    Uses the same visual style as plot_stress_tensor_comparison.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import jax.numpy as jnp
    
    os.makedirs(save_dir, exist_ok=True)
    eps = 1e-12  # small number to avoid divide-by-zero in relative error

    for idx in sample_indices:
        # Convert vector → 3x3 tensors (full matrix)
        L_true = np.array(vec9_to_square3(jnp.array([x_true_phys[idx]]))).squeeze()
        L_pred = np.array(vec9_to_square3(jnp.array([x_pred_phys[idx]]))).squeeze()
        L_err = L_true - L_pred
        L_relerr = np.abs(L_err) / (np.abs(L_true) + eps) * 100.0  # % relative error

        # Use same colormap scaling for True & Pred
        common_min = min(np.min(L_true), np.min(L_pred))
        common_max = max(np.max(L_true), np.max(L_pred))

        # Symmetric scale for absolute error colormap
        err_max = np.max(np.abs(L_err))

        # Create figure with 4 panels: True, Pred, Abs Error, Rel Error (%)
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))

        # --- True tensor ---
        im0 = axes[0].imshow(L_true, cmap="viridis", vmin=common_min, vmax=common_max)
        for (i, j), val in np.ndenumerate(L_true):
            axes[0].text(j, i, f"{val:.2e}", ha='center', va='center', color='white')
        axes[0].set_title(f"True (sample {idx})")
        fig.colorbar(im0, ax=axes[0], format="%.0e")

        # --- Predicted tensor ---
        im1 = axes[1].imshow(L_pred, cmap="viridis", vmin=common_min, vmax=common_max)
        for (i, j), val in np.ndenumerate(L_pred):
            axes[1].text(j, i, f"{val:.2e}", ha='center', va='center', color='white')
        axes[1].set_title(f"Predicted (sample {idx})")
        fig.colorbar(im1, ax=axes[1], format="%.0e")

        # --- Absolute Error tensor ---
        im2 = axes[2].imshow(L_err, cmap="RdBu_r", vmin=-err_max, vmax=err_max)
        for (i, j), val in np.ndenumerate(L_err):
            axes[2].text(j, i, f"{val:.2e}", ha='center', va='center', color='black')
        axes[2].set_title(f"Abs Error (True−Pred)")
        fig.colorbar(im2, ax=axes[2], format="%.0e")

        # --- Relative Error (% of True value) ---
        im3 = axes[3].imshow(L_relerr, cmap="inferno")
        for (i, j), val in np.ndenumerate(L_relerr):
            axes[3].text(j, i, f"{val:.2f}%", ha='center', va='center', color='white')
        axes[3].set_title("Relative Error (%)")
        fig.colorbar(im3, ax=axes[3], format="%.2f")

        plt.suptitle(f"{model_type} Velocity Gradient Tensor Comparison (sample {idx})", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"velocitygradient_tensor_comparison_with_error_sample_{idx}.png"))
        plt.close()






