import os
import numpy as np
import matplotlib.pyplot as plt
from utils.invariants import compute_invariants_vectorized

def write_sampledata_file(model_name, stability_status, L_list, T_list, condA_list, lam, save_root, cfg, X_flat, Y_flat, stage_tag):
    """
    Writes:
      1. First 3 samples diagnostics + combined L0/T plots + invariants diagram
      2. Unnormalized dataset statistics + combined boxplots & histograms
    """
    # ======================
    # 1) SAMPLE DATA ANALYSIS
    # ======================
    sample_dir = os.path.join(save_root, "sampledataanalysis")
    os.makedirs(sample_dir, exist_ok=True)

    # ---- Write samplesdata.txt ----
    filepath = os.path.join(sample_dir, "samplesdata.txt")
    with open(filepath, "w") as f:
        f.write(f"=== {stability_status} {model_name.replace('_', '-')} model ===\n")
        for idx in range(min(3, len(L_list))):
            L = L_list[idx]
            D = 0.5 * (L + L.T)
            A = np.eye(L.shape[0]) - lam * L
            T = T_list[idx]
            eigvals = np.linalg.eigvals(D)
            I_D, II_D, III_D = compute_invariants_vectorized(D)
            
            # ensure scalars
            I_D_scalar = np.array(I_D).item()
            II_D_scalar = np.array(II_D).item()
            III_D_scalar = np.array(III_D).item()
            
            discriminant_val = ((-III_D / 2)**2 + (-II_D / 3)**3)
            discriminant_val_scalar = np.array(discriminant_val).item()

            f.write(f"Sample {idx}:\n")
            f.write("L matrix:\n" + np.array2string(L, precision=4, suppress_small=True) + "\n")
            f.write(f"L norm = {np.linalg.norm(L):.4e}\n")
            f.write(f"Condition number(L) = {np.linalg.cond(L):.4f}\n\n")

            f.write("D matrix:\n" + np.array2string(D, precision=4, suppress_small=True) + "\n")
            f.write(f"D norm = {np.linalg.norm(D):.4e}\n")
            f.write(f"Condition number(D) = {np.linalg.cond(D):.4f}\n\n")
            f.write(f"Eigenvalues(D) = [{eigvals[0]:.4e}, {eigvals[1]:.4e}, {eigvals[2]:.4e}]\n")
            f.write(f"I_D = {I_D_scalar:.4e}, II_D = {II_D_scalar:.4e}, III_D = {III_D_scalar:.4e}\n")
            f.write(f"Discriminant(D) = {discriminant_val_scalar:.4e}\n\n")

            f.write("A matrix:\n" + np.array2string(A, precision=4, suppress_small=True) + "\n")
            f.write(f"A norm = {np.linalg.norm(A):.4e}\n")
            f.write(f"Condition number(A) = {np.linalg.cond(A):.4f}\n\n")

            f.write("T matrix:\n" + np.array2string(T, precision=4, suppress_small=True) + "\n")
            f.write(f"T norm = {np.linalg.norm(T):.4e}\n")
            f.write(f"T symmetry error Frobenius norm = {np.linalg.norm(T - T.T):.4e}\n")
            f.write(f"Condition number(T) = {np.linalg.cond(T):.4f}\n\n")
    print(f"✅ Sample data written to: {filepath}")
    
    # ---- Combined L0/T plots ----
    for idx in range(min(3, len(L_list))):
        L = L_list[idx]
        T = T_list[idx]
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Left subplot: L0
        im1 = axes[0].imshow(L, cmap='viridis')
        for (i, j), val in np.ndenumerate(L):
            axes[0].text(j, i, f"{val:.2e}", ha='center', va='center', color='w', fontsize=9)
        axes[0].set_title(f"L0 sample {idx}")
        fig.colorbar(im1, ax=axes[0])

        # Right subplot: T
        im2 = axes[1].imshow(T, cmap='viridis')
        for (i, j), val in np.ndenumerate(T):
            axes[1].text(j, i, f"{val:.2e}", ha='center', va='center', color='w', fontsize=9)
        axes[1].set_title(f"T sample {idx}")
        fig.colorbar(im2, ax=axes[1])

        fig.tight_layout()
        plt.savefig(os.path.join(sample_dir, f"sample_{idx}_L0_T_combined.png"), dpi=300)
        plt.close()
    print(f"✅ Combined L0+T plots saved to: {sample_dir}")

    # ---- Invariants diagram ----
    inv_path = os.path.join(sample_dir, f"{cfg.dim}D_InvariantsDiagram_{stability_status}.png")
    D0_list = [0.5 * (L + L.T) for L in L_list]
    D0_array = np.array(D0_list)
    plt.figure(figsize=(8, 6))
    II_vals, III_vals = np.linspace(0, 1.5, 300), np.linspace(-0.5, 0.5, 300)
    II_grid, III_grid = np.meshgrid(II_vals, III_vals)
    discriminant = ((-III_grid / 2)**2 + (-II_grid / 3)**3)

    II_boundary = np.linspace(0, 1.5, 500)
    III_boundary = 2 * np.sqrt(np.maximum(0, -(II_boundary / 3)**3))
    plt.plot(II_boundary, III_boundary, 'k-', linewidth=2)
    plt.plot(II_boundary, -III_boundary, 'k-', linewidth=2)

    plt.contourf(II_grid, III_grid, discriminant, levels=[-1e10, 0, 1e10],
                 colors=['#ffa8a8', '#a8c6ff'], alpha=0.8)
    
    # Scatter steady-state points
    _, II_D0, III_D0 = compute_invariants_vectorized(D0_array)
    plt.scatter(-II_D0, III_D0, color="red", s=40, edgecolors='k', linewidth=0.7)
    plt.title(f"{cfg.dim}D Invariants Diagram ({model_name}, stage {stage_tag})")
    plt.xlabel("-II_D")
    plt.ylabel("III_D")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(inv_path, dpi=300)
    plt.close()
    print(f"✅ Invariants diagram saved to: {inv_path}")
    
    # ======================
    # 2) UNNORMALIZED DATA ANALYSIS
    # ======================
    unnorm_dir = os.path.join(save_root, "unnormalized")
    os.makedirs(unnorm_dir, exist_ok=True)

    # ---- Stats file ----
    stats_path = os.path.join(unnorm_dir, "unnormalizeddatastat.txt")
    with open(stats_path, "w") as f:
        def write_stats(data_flat, var_name):
            mean_val = np.mean(data_flat)
            std_val  = np.std(data_flat)
            min_val, max_val = np.min(data_flat), np.max(data_flat)
            q1_val, median_val, q3_val = np.percentile(data_flat, [25, 50, 75])
            iqr_val = q3_val - q1_val
            f.write(f"--- {var_name} ---\n")
            f.write(f"  Shape:      {data_flat.shape}\n")
            f.write(f"  Count:      {data_flat.size}\n")
            f.write(f"  Mean:       {mean_val:.4e}\n")
            f.write(f"  Std Dev:    {std_val:.4e}\n")
            f.write(f"  Min:        {min_val:.4e}\n")
            f.write(f"  25% (Q1):   {q1_val:.4e}\n")
            f.write(f"  Median(Q2): {median_val:.4e}\n")
            f.write(f"  75% (Q3):   {q3_val:.4e}\n")
            f.write(f"  IQR:        {iqr_val:.4e}\n")
            f.write(f"  Max:        {max_val:.4e}\n\n")
            f.write(f"  Range:      {min_val:.4e}, {max_val:.4e}\n\n")
        write_stats(X_flat, "Velocity Gradient Tensor(L) - Unnormalized")
        write_stats(Y_flat, "Stress Tensor (T) - Unnormalized")

    # ---- Boxplots (combined figure) ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].boxplot(X_flat.flatten(), vert=True, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    axes[0].set_title(f"Boxplot - L (Unnormalized, stage {stage_tag})")
    axes[0].grid(True, ls="--", alpha=0.5)
    axes[1].boxplot(Y_flat.flatten(), vert=True, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
    axes[1].set_title(f"Boxplot - T (Unnormalized, stage {stage_tag})")
    axes[1].grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    plt.savefig(os.path.join(unnorm_dir, "boxplots_XY.png"), dpi=300)
    plt.close()

    # ---- Histograms (combined figure) ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].hist(X_flat.flatten(), bins=50, color='steelblue', edgecolor='white', log=True)
    axes[0].set_title(f"Histogram - L (Unnormalized, stage {stage_tag})")
    axes[0].grid(True, ls="--", alpha=0.5)
    axes[0].set_ylabel("Frequency")
    axes[1].hist(Y_flat.flatten(), bins=50, color='orange', edgecolor='white', log=True)
    axes[1].set_title(f"Histogram - T (Unnormalized, stage {stage_tag})")
    axes[1].grid(True, ls="--", alpha=0.5)
    axes[1].set_ylabel("Frequency")
    fig.tight_layout()
    plt.savefig(os.path.join(unnorm_dir, "histograms_XY.png"), dpi=300)
    plt.close()

    print(f"✅ Unnormalized dataset stats & plots saved to: {unnorm_dir}")