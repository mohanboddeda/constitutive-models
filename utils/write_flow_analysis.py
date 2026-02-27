import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from utils.invariants import compute_invariants_vectorized
from utils.matrixcelldata import add_text_with_contrast

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# Precise color mapping to match your reference plot exactly
FLOW_COLORS = {
    "uniaxial_extension": "blue",
    "biaxial_extension": "green",
    "planar_extension": "orange",
    "pure_shear": "red",
    "mixed_flow_above": "purple",
    "mixed_flow_below": "brown"
}

# Clean labels for the plot legends
FLOW_LABELS = {
    "uniaxial_extension": "Uniaxial ext.",
    "biaxial_extension": "Biaxial ext.",
    "planar_extension": "Planar ext.",
    "pure_shear": "Pure shear",
    "mixed_flow_above": "Mixed above axis",
    "mixed_flow_below": "Mixed below axis"
}

# =============================================================================
# PART 1: INDIVIDUAL FLOW ANALYSIS
# This function is called inside the loop for every single flow type.
# =============================================================================
def write_flow_analysis_file(
    model_name, flow_type, stability_status, L_list, T_list, condA_list, 
    lam, save_root, cfg, X_flat, Y_flat, stage_tag, residual_list=None
):
    """
    Generates comprehensive diagnostic plots and data logs for a SINGLE flow type.
    
    Parameters:
    -----------
    model_name : str
        Name of the Constitutive Model (e.g. "maxwell_B").
    flow_type : str
        Specific flow type (e.g. "uniaxial_extension").
    stability_status : str
        Mode of generation ("single_stage" or "multi_stage").
    L_list, T_list : list
        Lists of raw numpy matrices for Velocity Gradient and Stress.
    save_root : str
        Directory to save all images/logs.
    cfg : DictConfig
        Hydra configuration object.
    X_flat, Y_flat : np.ndarray
        Flattened arrays of all samples (for histograms/stats).
    stage_tag : str
        Label for the current stage (e.g. "1.0_2.4").
    """
    
    # Format names for titles
    clean_model_name = model_name.replace("_", " ").title()
    clean_flow_name = flow_type.replace("_", " ").title()
    
    # Get the specific color for this flow (default to black if unknown)
    flow_color = FLOW_COLORS.get(flow_type, "#333333") 

    # =========================================================================
    # 1. SAMPLE DATA LOGGING (Detailed Text Report)
    # =========================================================================
    sample_dir = os.path.join(save_root, "sampledataanalysis")
    os.makedirs(sample_dir, exist_ok=True)
    filepath = os.path.join(sample_dir, "samplesdata.txt")
    
    with open(filepath, "w") as f:
        f.write(f"=== {stability_status} {clean_model_name} | Flow: {clean_flow_name} ===\n")
        f.write(f"Stage Tag: {stage_tag}\n")
        f.write(f"Lambda: {lam}\n")
        f.write(f"Vorticity Ratio: {cfg.max_vorticity_ratio}\n\n")
        
        # Log first 3 samples for debugging
        for idx in range(min(3, len(L_list))):
            L = L_list[idx]
            D = 0.5 * (L + L.T)
            A = np.eye(L.shape[0]) - lam * L
            T = T_list[idx]
            eigvals = np.linalg.eigvals(D)
            I_D, II_D, III_D = compute_invariants_vectorized(D)
            
            # Ensure scalars
            I_D_s = np.array(I_D).item()
            II_D_s = np.array(II_D).item()
            III_D_s = np.array(III_D).item()
            
            f.write(f"--- Sample {idx} ---\n")
            f.write("L matrix:\n" + np.array2string(L, precision=4, suppress_small=True) + "\n")
            f.write(f"L norm = {np.linalg.norm(L):.4e}\n")
            f.write(f"Condition number(L) = {np.linalg.cond(L):.4f}\n\n")

            f.write("D matrix:\n" + np.array2string(D, precision=4, suppress_small=True) + "\n")
            f.write(f"Eigenvalues(D) = {np.array2string(eigvals, precision=4)}\n")
            f.write(f"Invariants: II={II_D_s:.4e}, III={III_D_s:.4e}\n\n")

            f.write("T matrix:\n" + np.array2string(T, precision=4, suppress_small=True) + "\n")
            f.write(f"T symmetry error = {np.linalg.norm(T - T.T):.4e}\n")
            f.write(f"Condition number(T) = {np.linalg.cond(T):.4f}\n\n")
            
    print(f"      ✅ Sample text data written to: {filepath}")

    # =========================================================================
    # 2. TENSOR HEATMAPS (Visual Inspection)
    # =========================================================================
    for idx in range(min(3, len(L_list))):
        L = L_list[idx]
        T = T_list[idx]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

        # Helper: Add Info Box
        def add_info_box(ax, flow, stage):
            text_str = f"Flow: {flow}\nStage: {stage}"
            props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
            ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='left', bbox=props)
        
        # Plot L
        im1 = axes[0].imshow(L, cmap='viridis')
        add_text_with_contrast(axes[0], L, "viridis", "{:.2e}")
        axes[0].set_title(fr"Velocity Gradient $\mathbf{{L}}$ (Sample {idx})", fontsize=14)
        axes[0].axis('off')
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        add_info_box(axes[0], clean_flow_name, stage_tag)

        # Plot T
        im2 = axes[1].imshow(T, cmap='viridis')
        add_text_with_contrast(axes[1], T, "viridis", "{:.2e}")
        axes[1].set_title(fr"Stress Tensor $\mathbf{{T}}$ (Sample {idx})", fontsize=14)
        axes[1].axis('off')
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        plt.savefig(os.path.join(sample_dir, f"sample_{idx}_L0_T_combined.png"), dpi=300)
        plt.close()

    print(f"      ✅ L & T Heatmaps saved.")

    # =========================================================================
    # 3. INDIVIDUAL INVARIANT DIAGRAM (Checking Geometry)
    # =========================================================================
    # FIX 1: Save inside 'sampledataanalysis' folder (sample_dir), not save_root
    inv_path = os.path.join(sample_dir, f"{cfg.dim}D_Flow_Distribution_Check.png")
    
    # Calculate Invariants from the list of tensors
    D0_list = [0.5 * (L + L.T) for L in L_list]
    D0_array = np.array(D0_list)
    _, neg_II, III = compute_invariants_vectorized(D0_array)
    neg_II = -neg_II # Convert to positive axis for plotting (-II_D)
    
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    
    # Define Lumley Triangle Boundaries
    x_max_plot = max(1.5, np.max(neg_II) * 1.1)
    x_b = np.linspace(0, x_max_plot, 1000)
    y_b = 2 * np.sqrt((x_b/3)**3)
    
    # Plot Background (Admissible Region)
    ax.set_facecolor('white')
    plt.fill_between(x_b, -y_b, y_b, color='lightblue', alpha=0.3, label='Admissible Region')
    plt.plot(x_b, y_b, 'k-', lw=1.5)
    plt.plot(x_b, -y_b, 'k-', lw=1.5)
    plt.axhline(0, color='k', lw=1.0, ls='--') 

    # Plot the Samples
    # FIX 2: Label format "Samples (Flow Name)"
    #sample_label = f"Samples ({clean_flow_name})"
    sample_label = f"Samples"
    plt.scatter(neg_II, III, c=flow_color, s=15, alpha=0.9, 
                edgecolors='none', label=sample_label, zorder=3)
    
    # Styling
    # FIX 3: Title format "Invariant Space for Flow Name"
    plt.title(f"Invariant Space for {clean_flow_name}", fontsize=14)
    plt.xlabel(r"Second Invariant ($-II_{\mathbf{D}}$)", fontsize=12)
    plt.ylabel(r"Third Invariant ($III_{\mathbf{D}}$)", fontsize=12)
    plt.xlim(0, x_max_plot)
    plt.ylim(-0.6, 0.6) # Standard vertical range
    
    # Legend
    # FIX 4: Add Model and Stage info to legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=sample_label,
               markerfacecolor=flow_color, markersize=8),
        Line2D([0], [0], color='none', label=f'Model: {clean_model_name}'),
        Line2D([0], [0], color='none', label=f'Stage: {stage_tag}'),
        Line2D([0], [0], color='none', label=f'Count: {len(L_list)}')
    ]
    plt.legend(handles=legend_elements, loc='upper left', frameon=True, framealpha=0.95)
    
    plt.grid(True, linestyle='-', alpha=0.5)
    plt.tight_layout()
    plt.savefig(inv_path, dpi=300)
    plt.close()
    
    print(f"      ✅ Invariants plot saved to {inv_path}")

    # =========================================================================
    # 4. UNNORMALIZED STATISTICS (Data Distribution)
    # =========================================================================
    unnorm_dir = os.path.join(save_root, "unnormalized")
    os.makedirs(unnorm_dir, exist_ok=True)
    
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
        
        f.write(f"FLOW TYPE: {clean_flow_name}\n\n")
        write_stats(X_flat, "Velocity Gradient Tensor (L)")
        write_stats(Y_flat, "Stress Tensor (T)")

    # =========================================================================
    # 5. BOXPLOTS (L & T)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].boxplot(X_flat.flatten(), vert=True, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    axes[0].set_title(f"Boxplot - L ({clean_flow_name})")
    axes[0].grid(True, ls="--", alpha=0.5)
    
    axes[1].boxplot(Y_flat.flatten(), vert=True, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
    axes[1].set_title(f"Boxplot - T ({clean_flow_name})")
    axes[1].grid(True, ls="--", alpha=0.5)
    
    fig.tight_layout()
    plt.savefig(os.path.join(unnorm_dir, "boxplots_XY.png"), dpi=300)
    plt.close()

    # =========================================================================
    # 6. HISTOGRAMS (L, T)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- L Histogram ---
    ax = axes[0]
    data = X_flat.flatten()
    ax.hist(data, bins=50, color='#4682B4', edgecolor='black', linewidth=0.5, log=True, alpha=0.9)
    ax.set_title(r"Velocity Gradient $\mathbf{L}$ Distribution", fontsize=14)
    ax.set_xlabel(r"Component Value ($L_{ij}$)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Stats Box L
    mu, sigma = np.mean(data), np.std(data)
    stats_text = f"$\\mu={mu:.2f}$\n$\\sigma={sigma:.2f}$"
    props = dict(boxstyle='round', facecolor='white', alpha=0.9)
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=11, ha='right', va='top', bbox=props)
    
    # Info Box
    model_info = f"Flow: {clean_flow_name}\nStage: {stage_tag}"
    ax.text(0.05, 0.95, model_info, transform=ax.transAxes, fontsize=10, ha='left', va='top', bbox=props)

    # --- T Histogram ---
    ax = axes[1]
    data = Y_flat.flatten()
    ax.hist(data, bins=50, color='#FF8C00', edgecolor='black', linewidth=0.5, log=True, alpha=0.9)
    ax.set_title(r"Stress Tensor $\mathbf{T}$ Distribution", fontsize=14)
    ax.set_xlabel(r"Component Value ($T_{ij}$)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)

    # Stats Box T
    mu, sigma = np.mean(data), np.std(data)
    stats_text = f"$\\mu={mu:.2f}$\n$\\sigma={sigma:.2f}$"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=11, ha='right', va='top', bbox=props)

    plt.tight_layout()
    plt.savefig(os.path.join(unnorm_dir, "histogram_XY.png"), dpi=300)
    plt.close()

    # =========================================================================
    # 7. HISTOGRAM D (Symmetric Part - Flow Specific Color)
    # =========================================================================
    # Reconstruct D from flat array
    X_mat = X_flat.reshape(-1, cfg.dim, cfg.dim)
    D = 0.5 * (X_mat + np.swapaxes(X_mat, 1, 2))
    
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    plt.hist(D.flatten(), bins=60, color=flow_color, 
             edgecolor='black', linewidth=0.5, log=True, alpha=0.8)
    
    plt.title(fr"Strain Rate $\mathbf{{D}}$ Distribution ({clean_flow_name})", fontsize=14)
    plt.xlabel("Component Value ($D_{ij}$)")
    plt.ylabel("Frequency (Log Scale)")
    plt.grid(True, alpha=0.3)
    
    # Stats Box D
    mu, sigma = np.mean(D.flatten()), np.std(D.flatten())
    stats_text = f"$\\mu={mu:.2f}$\n$\\sigma={sigma:.2f}$"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=11, ha='right', va='top', bbox=props)
    
    # Info Box
    ax.text(0.05, 0.95, model_info, transform=ax.transAxes, fontsize=10, ha='left', va='top', bbox=props)
    
    plt.tight_layout()
    hist_path = os.path.join(unnorm_dir, "histogram_D.png")
    plt.savefig(hist_path, dpi=300)
    plt.close()
    
    print(f"      ✅ Histograms saved to: {hist_path}")

    # =========================================================================
    # 8. RESIDUAL CHECK (Solver Verification)
    # =========================================================================
    if residual_list and len(residual_list) > 0:
        res_path = os.path.join(unnorm_dir, "solver_residuals.png")
        plt.figure(figsize=(10, 6))
        
        plt.hist(residual_list, bins=30, color='#663399', edgecolor='black', linewidth=0.5)
        plt.xscale('log')
        plt.xlim(1e-16, 1e-13) # Broaden axis
        
        plt.title(f"Solver Residuals ({clean_flow_name})", fontsize=16, pad=10)
        plt.xlabel(r"Residual Norm $||\mathbf{AT} + \mathbf{TB} - \mathbf{C}||_F$", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        
        # Stats
        mean_res = np.mean(residual_list)
        max_res = np.max(residual_list)
        stats_str = f"Mean = {mean_res:.2e}\n Max = {max_res:.2e}"
        
        plt.text(0.95, 0.95, stats_str, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right', 
                 bbox=dict(boxstyle='round', facecolor='white', edgecolor='#663399'))
        
        plt.grid(True, which="both", linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(res_path, dpi=300)
        plt.close()
        print(f"      ✅ Residual check plot saved.")


# =============================================================================
# PART 2: MERGED PLOTTING FUNCTION
# This function is called at the very end of the main script.
# It scans all subfolders and compiles the data into one Master Plot.
# =============================================================================
def plot_merged_flow_diagram(data_root_base, images_root_base, stage_tag):
    """
    Scans the folder structure for ALL flow types in the current stage
    and plots them on a SINGLE Combined Invariant Diagram.
    
    Parameters:
      data_root_base: Path to 'datafiles/.../stage_tag/'
      images_root_base: Path to 'images/.../stage_tag/'
      stage_tag: The name of the current stage (e.g. '1.0_2.4')
    """
    print(f"\n🎨 Generating Merged Invariant Plot for Stage: {stage_tag}...")
    
    # Setup Figure (Matches your reference style: Light Blue Background)
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    # 1. Draw Background (Lumley Triangle)
    # Fixed x-axis to 1.6 to match reference perspective
    x_b = np.linspace(0, 1.6, 1000)
    y_b = 2 * np.sqrt((x_b/3)**3)
    
    ax.set_facecolor('white') 
    plt.fill_between(x_b, y_b, -y_b, color='lightblue', alpha=0.3)
    
    # Draw boundaries
    plt.plot(x_b, y_b, 'k-', lw=2)
    plt.plot(x_b, -y_b, 'k-', lw=2)
    plt.axhline(0, color='gray', lw=1, ls='-')

    # 2. Iterate over known flow types and try to find data
    found_any = False
    
    # We iterate through the dictionary to ensure consistent color mapping
    for flow_key, flow_color in FLOW_COLORS.items():
        # Construct expected path: .../seed_X/stage_tag/flow_type/
        flow_folder = os.path.join(data_root_base, flow_key)
        
        # Skip if this flow type folder doesn't exist
        if not os.path.exists(flow_folder):
            continue
            
        # --- FIX: LOOK INSIDE SIZE FOLDERS ---
        # Old: glob.glob(os.path.join(flow_folder, "X_*.pt"))
        # New: Look into subdirectory "*" (e.g., "10ksamples" or "500samples")
        pt_files = glob.glob(os.path.join(flow_folder, "*", "X_*.pt"))
        
        if not pt_files:
            continue
            
        # Iterate over all found files (in case you have multiple runs/sizes)
        for pt_file in pt_files:
            try:
                # Load Data
                X_flat = torch.load(pt_file).numpy()
                X_mat = X_flat.reshape(-1, 3, 3) 
                D_mat = 0.5 * (X_mat + np.swapaxes(X_mat, 1, 2))
                
                # Compute Invariants
                _, neg_II, III = compute_invariants_vectorized(D_mat)
                neg_II = -neg_II # Convert to positive axis (-II_D)
                
                # Plot Points
                # Only label the first batch to avoid duplicate legend entries
                label_name = FLOW_LABELS.get(flow_key, flow_key) if not found_any else None
                # Actually, strictly only label ONCE per flow key
                if flow_key not in [l.get_label() for l in ax.get_lines() + ax.collections]:
                     curr_label = FLOW_LABELS.get(flow_key, flow_key)
                else:
                     curr_label = None

                plt.scatter(neg_II, III, c=flow_color, s=20, alpha=0.8, 
                            label=curr_label, edgecolors='none')
                
                found_any = True
                print(f"   -> Added {flow_key} ({len(neg_II)} points) from {os.path.basename(os.path.dirname(pt_file))}")
                
            except Exception as e:
                print(f"   -> Error loading {flow_key}: {e}")

    # 3. Finalize Plot
    if not found_any:
        print("   ⚠️ No data found to merge.")
        plt.close()
        return

    # Titles and Labels
    plt.title(f"Invariant Diagram – All Flow Types (Stage: {stage_tag})", fontsize=16)
    plt.xlabel(r"Second Invariant ($-II_{\mathbf{D}}$)", fontsize=14)
    plt.ylabel(r"Third Invariant ($III_{\mathbf{D}}$)", fontsize=14)
    
    # Axis Limits matching reference
    plt.xlim(0, 1.5)
    plt.ylim(-0.5, 0.5)
    
    # Grid
    plt.grid(True, linestyle='-', alpha=0.7)
    
    # Legend settings (Upper Left, white background)
    # Deduplicate legend labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', frameon=True, fontsize=10, 
               facecolor='white', framealpha=0.9, edgecolor='#ccc')
    
    # Save the merged plot in the base stage folder
    save_path = os.path.join(images_root_base, f"Merged_Invariant_Diagram_{stage_tag}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"✅ Merged plot saved to: {save_path}")