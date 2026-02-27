import os
import numpy as np
import matplotlib.pyplot as plt
from utils.invariants import compute_invariants_vectorized
from utils.matrixcelldata import add_text_with_contrast

def write_uniform_analysis_file(model_name, stability_status, L_list, T_list, condA_list, lam, save_root, cfg, X_flat, Y_flat, stage_tag, residual_list=None):
    """
    Writes:
      1. First 3 samples diagnostics + combined L0/T plots + invariants diagram (Uniformity Check)
      2. Unnormalized dataset statistics + combined boxplots & histograms
    
    Adapted for Uniform Data Generation to ensure consistency with write_sampledata1.py.
    """
    # ======================
    # 1) SAMPLE DATA ANALYSIS
    # ======================
    sample_dir = os.path.join(save_root, "sampledataanalysis")
    os.makedirs(sample_dir, exist_ok=True)

    # ---- Write samplesdata.txt ----
    filepath = os.path.join(sample_dir, "samplesdata.txt")
    with open(filepath, "w") as f:
        f.write(f"=== {stability_status} {model_name.replace('_', '-')} model (Uniform Generation) ===\n")
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
            
            # For uniform data, we expect points to fill the triangle, not cluster
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
    
    # =================================================================
    # IMPROVED TENSOR HEATMAPS (Clean - No Axis Labels)
    # =================================================================
    for idx in range(min(3, len(L_list))):
        L = L_list[idx]
        T = T_list[idx]
        
        # Use constrained_layout for auto-spacing
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

        # --- Helper: Info Box (Safety Check) ---
        if 'add_info_box' not in locals():
            def add_info_box(ax, model_name, stage_name):
                text_str = f"Model: {model_name}\nStage: {stage_name}"
                props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
                ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', horizontalalignment='left', bbox=props)
        
        if 'clean_model_name' not in locals():
            clean_model_name = model_name.replace("_", " ").title()

        # --- Left Subplot: Velocity Gradient L ---
        im1 = axes[0].imshow(L, cmap='viridis')
        add_text_with_contrast(axes[0], L, "viridis", "{:.2e}")
        axes[0].set_title(fr"Velocity Gradient $\mathbf{{L}}$ (Sample {idx})", fontsize=14)
        
        # REMOVE TICKS (The numbers 0, 1, 2 on the sides)
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        # Colorbar
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Add Info Box
        add_info_box(axes[0], clean_model_name, stage_tag)

        # --- Right Subplot: Stress Tensor T ---
        im2 = axes[1].imshow(T, cmap='viridis')
        add_text_with_contrast(axes[1], T, "viridis", "{:.2e}")
        axes[1].set_title(fr"Stress Tensor $\mathbf{{T}}$ (Sample {idx})", fontsize=14)
        
        # REMOVE TICKS
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        # Colorbar
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        # Save
        plt.savefig(os.path.join(sample_dir, f"sample_{idx}_L0_T_combined.png"), dpi=300)
        plt.close()

    print(f"✅ L0+T plots saved to: {sample_dir}")

    # =================================================================
    # 2. INVARIANTS DIAGRAM (Clean White Background + Aligned Info)
    # =================================================================
    inv_path = os.path.join(sample_dir, f"{cfg.dim}D_Uniform_Distribution_Check.png")
    
    # Calculate Invariants
    D0_list = [0.5 * (L + L.T) for L in L_list]
    D0_array = np.array(D0_list)
    _, neg_II, III = compute_invariants_vectorized(D0_array)
    neg_II = -neg_II # Convert to positive magnitude (-II_D)
    
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    
    # --- Boundaries of Lumley Triangle ---
    x_b = np.linspace(0, 1.6, 1000)
    y_b = 2 * np.sqrt((x_b/3)**3)
    
    # --- Plotting Layers ---
    
    # 1. Background: Default White
    ax.set_facecolor('#ffa8a8')
    
    # 2. Admissible Region (Blue Triangle)
    plt.fill_between(x_b, -y_b, y_b, color='#a8c6ff', alpha=1.0, zorder=0)
    
    # 3. Boundary Lines
    plt.plot(x_b, y_b, 'k-', lw=1.5, zorder=1)
    plt.plot(x_b, -y_b, 'k-', lw=1.5, zorder=1)
    plt.axhline(0, color='k', lw=1.5, zorder=1)
    
    # 4. Points (High Contrast Red)
    plt.scatter(neg_II, III, c='#b30000', s=12, alpha=0.9, 
                edgecolors='none', zorder=2)
    
    # --- Styling ---
    plt.title("Invariant Space Distribution", fontsize=16, pad=10)
    plt.xlabel(r"Second Invariant ($-II_{\mathbf{D}}$)", fontsize=14)
    plt.ylabel(r"Third Invariant ($III_{\mathbf{D}}$)", fontsize=14)
    
    # --- Legend & Info Box (Neatly Aligned) ---
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    clean_model_name = model_name.replace("_", " ").title()
    
    # Construct handles for the legend
    legend_elements = [
        # Visual Keys
        Patch(facecolor='#d0e0ff', edgecolor='none', label='Admissible Region'),
        Line2D([0], [0], marker='o', color='w', label='Generated Samples',
               markerfacecolor='#b30000', markersize=8),
        
        # Spacer (Empty invisible line)
        Line2D([0], [0], color='none', label=''), 
        
        # Text Info (Aligned neatly below)
        Line2D([0], [0], color='none', label=f'Model: {clean_model_name}'),
        Line2D([0], [0], color='none', label=f'Stage: {stage_tag}'),
        Line2D([0], [0], color='none', label=f'Count: {len(L_list)} samples'),
        #Line2D([0], [0], color='none', label=f'Mode:  {stability_status}')
    ]

    # Create single box top-left
    # handlelength/handletextpad ensures text lines align with visual lines
    ax.legend(handles=legend_elements, loc='upper left', 
              frameon=True, framealpha=0.95, edgecolor='#cccccc',
              fontsize=10, borderpad=1.0, labelspacing=0.4,
              handlelength=1.5, handletextpad=0.5)

    plt.xlim(0, 1.5)
    plt.ylim(-0.6, 0.6)
    
    # Faint grid
    plt.grid(True, linestyle=':', alpha=0.4, color='gray', zorder=0)
    
    plt.tight_layout()
    plt.savefig(inv_path, dpi=300)
    plt.close()
    
    print(f"✅ Invariants plot saved to {inv_path}")
    
    # ====================================
    # 4) UNNORMALIZED DATA ANALYSIS
    # ====================================
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
    axes[0].set_title(f"Boxplot - L (Unnormalized, {model_name}, stage {stage_tag})")
    axes[0].grid(True, ls="--", alpha=0.5)
    axes[1].boxplot(Y_flat.flatten(), vert=True, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
    axes[1].set_title(f"Boxplot - T (Unnormalized, {model_name}, stage {stage_tag})")
    axes[1].grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    plt.savefig(os.path.join(unnorm_dir, "boxplots_XY.png"), dpi=300)
    plt.close()

    # =================================================================
    # 3. HISTOGRAMS (Fixed: Info Box on Right & No "Type: Uniform")
    # =================================================================
    hist_path = os.path.join(unnorm_dir, "histogram_XY.png")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Plot L (Velocity Gradient) ---
    ax = axes[0]
    data = X_flat.flatten()
    ax.hist(data, bins=50, color='#4682B4', edgecolor='black', linewidth=0.5, log=True, alpha=0.9)
    ax.set_title(r"Distribution of Velocity Gradient Tensor $\mathbf{L}$", fontsize=14)
    ax.set_xlabel(r"Component Value ($L_{ij}$)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.grid(True, which="both", linestyle='--', alpha=0.3)
    
    # Statistics for L (Top Right)
    mu = np.mean(data)
    sigma = np.std(data)
    # Using double backslash to fix syntax warnings
    stats_text = f"$\\mu={mu:.3f}$\n$\\sigma={sigma:.3f}$"
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9)
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    # Info Box (MOVED TO RIGHT, STACKED BELOW STATS)
    # Removed "\nType: Uniform"
    model_info = f"Model: {model_name.replace('_', ' ').title()}\nStage: {stage_tag}"
    
    ax.text(0.05, 0.95, model_info, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left', bbox=props)

    # --- Plot T (Stress Tensor) ---
    ax = axes[1]
    data = Y_flat.flatten()
    ax.hist(data, bins=50, color='#FF8C00', edgecolor='black', linewidth=0.5, log=True, alpha=0.9)
    ax.set_title(r"Distribution of Stress Tensor $\mathbf{T}$", fontsize=14)
    ax.set_xlabel(r"Component Value ($T_{ij}$)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.grid(True, which="both", linestyle='--', alpha=0.3)

    # Statistics for T (Top Right)
    mu = np.mean(data)
    sigma = np.std(data)
    stats_text = f"$\\mu={mu:.3f}$\n$\\sigma={sigma:.3f}$"
    
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right', bbox=props) 

    # Info Box (MOVED TO RIGHT, STACKED BELOW STATS)
    ax.text(0.95, 0.82, model_info, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"✅ Histograms saved to: {hist_path}")

    # =================================================================
    # 6) IMPROVED HISTOGRAM - D (Symmetric Part)
    # =================================================================
    
    # Prepare D matrix
    X_mat = X_flat.reshape(-1, cfg.dim, cfg.dim)
    D = 0.5 * (X_mat + np.swapaxes(X_mat, 1, 2))

    # Setup Figure
    plt.figure(figsize=(10, 6))
    ax = plt.gca()  # Get current axis for adding boxes

    # Ensure helper functions exist (in case you didn't run the previous block)
    if 'add_stats_box' not in locals():
        def add_stats_box(ax, data, color):
            mean_val, std_val = np.mean(data), np.std(data)
            text_str = '\n'.join((r'$\mu=%.3f$' % mean_val, r'$\sigma=%.3f$' % std_val))
            props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=color)
            ax.text(0.95, 0.95, text_str, transform=ax.transAxes, fontsize=11,
                    verticalalignment='top', horizontalalignment='right', bbox=props)
    
    if 'add_info_box' not in locals():
        def add_info_box(ax, model_name, stage_name):
            text_str = f"Model: {model_name}\nStage: {stage_name}"
            props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
            ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='left', bbox=props)
            
    # Ensure clean model name exists
    if 'clean_model_name' not in locals():
        clean_model_name = model_name.replace("_", " ").title()

    # --- Plot D ---
    _ = plt.hist(D.flatten(), bins=60, color='#228B22', 
             edgecolor='black', linewidth=0.5, log=True, alpha=1.0)

    # Titles and Labels
    plt.title(fr"Distribution of Strain-rate tensor $\mathbf{{D}}$", fontsize=14, pad=10)
    plt.xlabel(r"Component Value ($D_{ij}$)", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.3)

    # Add Info Boxes (Top Left & Top Right)
    add_stats_box(ax, D.flatten(), '#228B22')
    add_info_box(ax, clean_model_name, stage_tag)

    # Save
    plt.tight_layout()
    save_path_D = os.path.join(unnorm_dir, "histogram_D.png")
    plt.savefig(save_path_D, dpi=300)
    plt.close()

    print(f"✅ Histograms saved to: {save_path_D}")

   # =================================================================
    # 4. RESIDUAL CHECK (Original Layout + Broader Axis)
    # =================================================================
    if residual_list and len(residual_list) > 0:
        res_path = os.path.join(unnorm_dir, "solver_residuals.png")
        plt.figure(figsize=(10, 6))
        
        # Plot Histogram
        plt.hist(residual_list, bins=30, color='#663399', edgecolor='black', linewidth=0.5)
        
        plt.xscale('log')
        # --- FIX: Broaden the X-Axis ---
        plt.xlim(1e-16, 1e-13)
        
        plt.title("Solver Residual Distribution", fontsize=16, pad=10)
        plt.xlabel(r"Residual Norm $||\mathbf{AT} + \mathbf{TB} - \mathbf{C}||_F$", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        
        # Statistics
        mean_res = np.mean(residual_list)
        max_res = np.max(residual_list)
        stats_str = f"Mean = {mean_res:.2e}\n Max = {max_res:.2e}"
        
        # Info Box (Left - Original Position) & Stats (Right)
        props = dict(boxstyle='round', facecolor='white', alpha=0.9)
        plt.text(0.95, 0.95, stats_str, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', edgecolor='#663399'))
        
        # REMOVED: "\nType: Uniform"
        model_info = f"Model: {model_name.replace('_', ' ').title()}\nStage: {stage_tag}"
        plt.text(0.05, 0.95, model_info, transform=plt.gca().transAxes, fontsize=11,
                 verticalalignment='top', horizontalalignment='left', bbox=props)
        
        plt.grid(True, which="both", linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(res_path, dpi=300)
        plt.close()
        print(f"✅ Residual check plot saved to: {res_path}")