import os
import numpy as np
import matplotlib.pyplot as plt
from utils.invariants import compute_invariants_vectorized
from utils.matrixcelldata import add_text_with_contrast
 # --- AXIS FORMATTING FIX ---
import matplotlib.ticker as ticker
 # --- Merged Legend & Info Box (Cleaned) ---
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def write_sampledata_file(model_name, stability_status, L_list, T_list, condA_list, lam, save_root, cfg, X_flat, Y_flat, stage_tag, residual_list=None):
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
    # INVARIANTS DIAGRAM (Corrected: Full Background & Clean Legend)
    # =================================================================
    inv_path = os.path.join(sample_dir, f"{cfg.dim}D_InvariantsDiagram_{stability_status}.png")
    
    # --- Data Preparation ---
    D0_list = [0.5 * (L + L.T) for L in L_list]
    D0_array = np.array(D0_list)

    # --- Plot Setup ---
    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    # --- Helper: Clean Model Name ---
    if 'clean_model_name' not in locals():
        clean_model_name = model_name.replace("_", " ").title()

    # --- Background (Fixed to fill entire plot) ---
    # Extended III range to +/- 0.7 so it covers the ylim of +/- 0.6
    II_vals = np.linspace(0, 1.6, 400)
    III_vals = np.linspace(-0.7, 0.7, 400) 
    II_grid, III_grid = np.meshgrid(II_vals, III_vals)
    discriminant = ((-III_grid / 2)**2 + (-II_grid / 3)**3)

    # Colors: Inside=#ffa8a8 (Pink), Outside=#a8c6ff (Blue)
    plt.contourf(II_grid, III_grid, discriminant, levels=[-1e10, 0, 1e10],
                 colors=['#ffa8a8', '#a8c6ff'], alpha=0.8, zorder=0)

    # --- Curves (Consistent Style) ---
    x_b = np.linspace(0, 1.6, 1000)
    y_b = 2 * np.sqrt((x_b/3)**3)
    
    plt.plot(x_b, y_b, 'k-', lw=1.5, zorder=1)
    plt.plot(x_b, -y_b, 'k-', lw=1.5, zorder=1)
    plt.axhline(0, color='k', lw=1.5, zorder=1)

    # --- Scatter Points ---
    _, II_D0, III_D0 = compute_invariants_vectorized(D0_array)
    plt.scatter(-II_D0, III_D0, c='red', s=40, edgecolors='k', linewidth=0.7, zorder=2)

    # --- Styling ---
    plt.title("Invariant Space Distribution", fontsize=16, pad=10)
    plt.xlabel(r"Second Invariant ($-II_{\mathbf{D}}$)", fontsize=14)
    plt.ylabel(r"Third Invariant ($III_{\mathbf{D}}$)", fontsize=14)
    
    # Limits
    plt.xlim(0, 1.5)
    plt.ylim(-0.6, 0.6)

    legend_elements = [
        # Visual Keys (Only Admissible & Points, removed Unadmissible text)
        Patch(facecolor='#ffa8a8', edgecolor='none', label='Admissible Region'),
        Line2D([0], [0], marker='o', color='w', label='Generated Samples',
               markerfacecolor='red', markersize=8, markeredgecolor='k'),
        
        # Spacer
        Line2D([0], [0], color='none', label=''), 
        
        # Dataset Info
        Line2D([0], [0], color='none', label=f'Model: {clean_model_name}'),
        Line2D([0], [0], color='none', label=f'Stage: {stage_tag}'),
        Line2D([0], [0], color='none', label=f'Count: {len(L_list)} samples'),
        #Line2D([0], [0], color='none', label=f'Mode:  {stability_status}')
    ]

    # Create the legend box
    ax.legend(handles=legend_elements, loc='upper left', 
              frameon=True, framealpha=0.95, edgecolor='#cccccc',
              fontsize=10, borderpad=1.0, labelspacing=0.4,
              handlelength=1.5, handletextpad=0.5)

    plt.grid(True, linestyle='--', alpha=0.4, zorder=0)
    plt.tight_layout()
    plt.savefig(inv_path, dpi=300)
    plt.close()
    
    print(f"✅ Invariants diagram saved to: {inv_path}")
    
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
    # 5) IMPROVED HISTOGRAMS of L and T
    # =================================================================
    
    # Optional: Set a cleaner font style for the plot
    plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # --- Helper 1: Stats Box (Top Right) ---
    def add_stats_box(ax, data, color):
        mean_val = np.mean(data)
        std_val = np.std(data)
        text_str = '\n'.join((
            r'$\mu=%.3f$' % (mean_val, ),
            r'$\sigma=%.3f$' % (std_val, )))
        
        # Place text in top right (x=0.95, y=0.95)
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=color)
        ax.text(0.95, 0.95, text_str, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right', bbox=props)

    # --- Helper 2: Info Box (Top Left) ---
    def add_info_box(ax, model_name, stage_name):
        # Create the text string
        text_str = f"Model: {model_name}\nStage: {stage_name}"
        
        # Place text in top left (x=0.05, y=0.95)
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='left', bbox=props)

    # Clean up model name (e.g., "maxwell_B" -> "Maxwell B")
    clean_model_name = model_name.replace("_", " ").title()

    # --- Plot 1: Velocity Gradient L ---
    _ = axes[0].hist(X_flat.flatten(), bins=60, color='#4682B4', 
                 edgecolor='black', linewidth=0.5, log=True, alpha=1.0)
    
    # Main Title (Clean and Descriptive)
    axes[0].set_title(fr"Distribution of Velocity Gradient Tensor $\mathbf{{L}}$", fontsize=14, pad=10)
    axes[0].set_xlabel(r"Component Value ($L_{ij}$)", fontsize=14)
    axes[0].set_ylabel("Frequency", fontsize=14)
    axes[0].grid(True, which="both", ls="--", alpha=0.3)
    
    # Add Boxes
    add_stats_box(axes[0], X_flat.flatten(), '#4682B4')     # Top Right
    add_info_box(axes[0], clean_model_name, stage_tag)      # Top Left

    # --- Plot 2: Stress Tensor T ---
    _ = axes[1].hist(Y_flat.flatten(), bins=60, color='#FF8C00', 
                 edgecolor='black', linewidth=0.5, log=True, alpha=1.0)
    
    # Main Title
    axes[1].set_title(fr"Distribution of Stress Tensor $\mathbf{{T}}$", fontsize=14, pad=10)
    axes[1].set_xlabel(r"Component Value ($T_{ij}$)", fontsize=14)
    axes[1].set_ylabel("Frequency", fontsize=14)
    axes[1].grid(True, which="both", ls="--", alpha=0.3)
    
    # Add Boxes
    add_stats_box(axes[1], Y_flat.flatten(), '#FF8C00')     # Top Right
    add_info_box(axes[1], clean_model_name, stage_tag)      # Top Left

    # --- Save Figure ---
    save_path = os.path.join(unnorm_dir, "histograms_XY.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"✅ Unnormalized dataset stats & plots saved to: {unnorm_dir}")

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

    print(f"✅ Histograms saved to: {save_path}")

    # =================================================================
    # 7) SOLVER RESIDUAL CHECK PLOT
    # =================================================================
    if residual_list is not None and len(residual_list) > 0:
        residuals = np.array(residual_list)
        
        # Log10 for histogram
        log_residuals = np.log10(residuals + 1e-30)

        plt.figure(figsize=(9, 6))
        ax = plt.gca()

        # --- Helper for Info Box ---
        if 'add_info_box' not in locals():
            def add_info_box(ax, model_name, stage_name):
                text_str = f"Model: {model_name}\nStage: {stage_name}"
                props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
                ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', horizontalalignment='left', bbox=props)
        
        if 'clean_model_name' not in locals():
            clean_model_name = model_name.replace("_", " ").title()

        # --- Plot Histogram ---
        plt.hist(log_residuals, bins=30, color='#663399', 
         edgecolor='black', linewidth=0.5, alpha=1.0)

        # 1. Force ticks to be integers only (-16, -15, -14)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        
        # 2. Format them as powers of 10
        def log_tick_formatter(x, pos):
            return r"$10^{%d}$" % int(x)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(log_tick_formatter))

        # --- Titles & Labels ---
        plt.title("Solver Residual Distribution", fontsize=14, pad=10)
        plt.xlabel(r"Residual Norm $|| \mathbf{A}\mathbf{T} + \mathbf{T}\mathbf{B} - \mathbf{C} ||_F$", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.grid(True, ls="--", alpha=0.3)

        # --- Add Boxes ---
        add_info_box(ax, clean_model_name, stage_tag)

        mean_res = np.mean(residuals)
        max_res = np.max(residuals)
        text_str = '\n'.join((
            r'Mean $= %.2e$' % (mean_res, ),
            r'Max $= %.2e$' % (max_res, )))
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#663399')
        ax.text(0.95, 0.95, text_str, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        
        # Save
        res_path = os.path.join(unnorm_dir, "solver_residuals.png")
        plt.tight_layout()
        plt.savefig(res_path, dpi=300)
        plt.close()
        print(f"✅ Residual check plot saved to: {res_path}")