import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_stage_progression(
    data_root,
    images_root,
    model_name="maxwell_B",
    suffix="_stage",
    stages=None,
    cmap_name="coolwarm",
    n_samples=None   # <--- NEW ARGUMENT
):
    """
    Plots L, D, and T histogram progression using a sequential colormap.
    Uses density=True to ensure curves are comparable regardless of sample count.
    Now supports sub-folders based on n_samples (e.g., '10ksamples').
    """
    if stages is None:
        raise ValueError("Please pass a list of stage names.")

    os.makedirs(images_root, exist_ok=True)
    
    # --- SETUP PROFESSIONAL STYLE ---
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    
    # Create a gradient of colors
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, len(stages)))

    clean_model_name = model_name.replace("_", " ").title()

    # --- HELPER: Determine Size Folder ---
    size_folder = ""
    if n_samples is not None:
        if n_samples >= 1000:
            size_folder = f"{int(n_samples/1000)}ksamples"
        else:
            size_folder = f"{n_samples}samples"

    # =================================================================
    # 1. Velocity Gradient (L) Progression
    # =================================================================
    plt.figure(figsize=(8, 6))
    
    found_any = False
    for i, stage_tag in enumerate(stages):
        # UPDATED PATH LOGIC
        X_path = os.path.join(data_root, stage_tag, size_folder, f"X_3D_{model_name}{suffix}.pt")
        
        if not os.path.exists(X_path):
            print(f"⚠️ Skipping {stage_tag}, file not found: {X_path}")
            continue
        
        found_any = True
        X = torch.load(X_path).numpy().flatten()
        
        # density=True means the area under curve is 1.0 (Probability Density)
        plt.hist(X, bins=100, alpha=0.8, color=colors[i], label=stage_tag,
                 density=True, histtype='step', linewidth=2.0)

    if found_any:
        plt.title(fr"Evolution of Velocity Gradient $\mathbf{{L}}$ ({clean_model_name})", fontsize=16, pad=10)
        plt.xlabel(r"Component Value ($L_{ij}$)", fontsize=14)
        plt.ylabel("Probability Density", fontsize=14)
        plt.yscale('log')
        
        plt.legend(frameon=True, fontsize=10, ncol=2, loc='upper right', framealpha=0.95)
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.tight_layout()
        
        save_path_L = os.path.join(images_root, f"L_distribution_progression_{model_name}{suffix}.png")
        plt.savefig(save_path_L, dpi=300)
        print(f"   📈 Saved L-Progression to {save_path_L}")
    plt.close()


    # =================================================================
    # 2. Rate-of-Deformation (D) Progression
    # =================================================================
    plt.figure(figsize=(8, 6))
    
    found_any = False
    for i, stage_tag in enumerate(stages):
        X_path = os.path.join(data_root, stage_tag, size_folder, f"X_3D_{model_name}{suffix}.pt")
        if not os.path.exists(X_path):
            continue
            
        found_any = True
        X = torch.load(X_path).numpy().reshape(-1, 3, 3)
        D = 0.5 * (X + np.swapaxes(X, 1, 2))
        
        plt.hist(D.flatten(), bins=100, alpha=0.8, color=colors[i], label=stage_tag,
                 density=True, histtype='step', linewidth=2.0)

    if found_any:
        plt.title(fr"Evolution of Rate-of-Deformation $\mathbf{{D}}$ ({clean_model_name})", fontsize=16, pad=10)
        plt.xlabel(r"Component Value ($D_{ij}$)", fontsize=14)
        plt.ylabel("Probability Density", fontsize=14)
        plt.yscale('log')
        plt.legend(frameon=True, fontsize=10, ncol=2, loc='upper right', framealpha=0.95)
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.tight_layout()
        
        save_path_D = os.path.join(images_root, f"D_distribution_progression_{model_name}{suffix}.png")
        plt.savefig(save_path_D, dpi=300)
        print(f"   📈 Saved D-Progression to {save_path_D}")
    plt.close()


    # =================================================================
    # 3. Stress Tensor (T) Progression
    # =================================================================
    plt.figure(figsize=(8, 6))
    
    found_any = False
    for i, stage_tag in enumerate(stages):
        Y_path = os.path.join(data_root, stage_tag, size_folder, f"Y_3D_{model_name}{suffix}.pt")
        if not os.path.exists(Y_path):
            continue
            
        found_any = True
        Y = torch.load(Y_path).numpy().flatten()
        
        plt.hist(Y, bins=100, alpha=0.8, color=colors[i], label=stage_tag,
                 density=True, histtype='step', linewidth=2.0)

    if found_any:
        plt.title(fr"Evolution of Stress Tensor $\mathbf{{T}}$ ({clean_model_name})", fontsize=16, pad=10)
        plt.xlabel(r"Component Value ($T_{ij}$)", fontsize=14)
        plt.ylabel("Probability Density", fontsize=14)
        plt.yscale('log')
        plt.legend(frameon=True, fontsize=10, ncol=2, loc='upper right', framealpha=0.95)
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.tight_layout()
        
        save_path_T = os.path.join(images_root, f"T_distribution_progression_{model_name}{suffix}.png")
        plt.savefig(save_path_T, dpi=300)
        print(f"   📈 Saved T-Progression to {save_path_T}")
    plt.close()
    
    print(f"✅ Progression plots saved using '{cmap_name}' colormap.")