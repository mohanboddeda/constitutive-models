import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_stage_progression(
    data_root,
    images_root,
    model_name="maxwell_B",
    suffix="_stable",
    stages=None,
    cmap_name="tab10"
):
    """
    Plots L, D, and T histogram progression over all stages (combined)
    and also saves stage-wise histograms in each stage's 'sampledataanalysis' folder.
    """
    if stages is None:
        raise ValueError("Please pass a list of stage names.")

    os.makedirs(images_root, exist_ok=True)
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i % cmap.N) for i in range(len(stages))]

    # ---------- Combined progression plots ----------
    # L progression
    plt.figure(figsize=(7, 5))
    for i, stage_tag in enumerate(stages):
        X_path = os.path.join(data_root, stage_tag, f"X_3D_{model_name}{suffix}.pt")
        if not os.path.exists(X_path):
            print(f"[WARN] Missing file: {X_path}")
            continue
        X = torch.load(X_path).numpy().flatten()
        plt.hist(X, bins=100, alpha=0.6, color=colors[i], label=stage_tag,
                 density=True, histtype='step', linewidth=1.5)
    plt.title("L (Velocity Gradient) Distribution Progression", fontsize=14)
    plt.xlabel("L values")
    plt.ylabel("Density")
    plt.yscale('log')
    plt.legend(frameon=False, fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(images_root, f"L_distribution_progression_{model_name}{suffix}.png"), dpi=600)
    plt.close()

    # D progression
    plt.figure(figsize=(7, 5))
    for i, stage_tag in enumerate(stages):
        X_path = os.path.join(data_root, stage_tag, f"X_3D_{model_name}{suffix}.pt")
        if not os.path.exists(X_path):
            continue
        X = torch.load(X_path).numpy().reshape(-1, 3, 3)
        D = 0.5 * (X + np.swapaxes(X, 1, 2))
        plt.hist(D.flatten(), bins=100, alpha=0.6, color=colors[i], label=stage_tag,
                 density=True, histtype='step', linewidth=1.5)
    plt.title("D (Rate-of-Deformation) Distribution Progression", fontsize=14)
    plt.xlabel("D values")
    plt.ylabel("Density")
    plt.yscale('log')
    plt.legend(frameon=False, fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(images_root, f"D_distribution_progression_{model_name}{suffix}.png"), dpi=600)
    plt.close()

    # T progression
    plt.figure(figsize=(7, 5))
    for i, stage_tag in enumerate(stages):
        Y_path = os.path.join(data_root, stage_tag, f"Y_3D_{model_name}{suffix}.pt")
        if not os.path.exists(Y_path):
            continue
        Y = torch.load(Y_path).numpy().flatten()
        plt.hist(Y, bins=100, alpha=0.6, color=colors[i], label=stage_tag,
                 density=True, histtype='step', linewidth=1.5)
    plt.title("T (Stress Tensor) Distribution Progression", fontsize=14)
    plt.xlabel("T values")
    plt.ylabel("Density")
    plt.yscale('log')
    plt.legend(frameon=False, fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(images_root, f"T_distribution_progression_{model_name}{suffix}.png"), dpi=600)
    plt.close()

    # ---------- Stage-wise histograms ----------
    for stage_tag in stages:
        stage_analysis_dir = os.path.join(images_root, f"{stage_tag} stable", model_name, "sampledataanalysis")
        os.makedirs(stage_analysis_dir, exist_ok=True)

        # Load stage data
        X_path = os.path.join(data_root, stage_tag, f"X_3D_{model_name}{suffix}.pt")
        Y_path = os.path.join(data_root, stage_tag, f"Y_3D_{model_name}{suffix}.pt")
        if not (os.path.exists(X_path) and os.path.exists(Y_path)):
            print(f"[WARN] Missing L/T data for stage {stage_tag}")
            continue

        X = torch.load(X_path).numpy()
        Y = torch.load(Y_path).numpy()

        # ---- D distribution plot ----
        X_mat = X.reshape(-1, 3, 3)
        D = 0.5 * (X_mat + np.swapaxes(X_mat, 1, 2))
        plt.figure(figsize=(5, 4))
        plt.hist(D.flatten(), bins=100, color="green", alpha=0.7, density=True)
        plt.title(f"D Distribution ({stage_tag})")
        plt.xlabel("D values")
        plt.ylabel("Density")
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(stage_analysis_dir, f"{stage_tag}_D_hist.png"), dpi=600)
        plt.close()

        print(f"✅ Saved stage-wise plots for {stage_tag} in {stage_analysis_dir}")