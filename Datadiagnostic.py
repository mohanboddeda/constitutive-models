# Datadiagnostic.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from numpy.linalg import cond
from utils.invariants import compute_invariants_vectorized
from TensorJaxReplay import vec6_to_sym3

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="config/data", config_name="dataConfig", version_base=None)
def main(cfg: DictConfig):
    np.random.seed(cfg.seed)
    mode = "stable" if cfg.stable else "unstable"

    print(f"\n=== Processing {mode.upper()} data for model {cfg.constitutive_eq} ===")

    if mode == "stable":
        # stage definitions
        stages = [
            {"stage_tag": "1.0"}, {"stage_tag": "1.0_1.2"}, {"stage_tag": "1.2_1.4"},
            {"stage_tag": "1.4_1.6"}, {"stage_tag": "1.6_1.8"}, {"stage_tag": "1.8_2.0"},
            {"stage_tag": "2.0_2.2"}, {"stage_tag": "2.2_2.4"}, {"stage_tag": "2.4_2.6"},
            {"stage_tag": "2.6_2.8"}, {"stage_tag": "2.8_3.0"},
        ]
        shear_means, stress_means, cond_means, stage_tags = [], [], [], []

        for s in stages:
            tag = s["stage_tag"]
            X_path = os.path.join(cfg.paths.data, tag, f"X_{cfg.dim}D_{cfg.constitutive_eq}_stable.pt")
            Y_path = os.path.join(cfg.paths.data, tag, f"Y_{cfg.dim}D_{cfg.constitutive_eq}_stable.pt")
            if not (os.path.exists(X_path) and os.path.exists(Y_path)):
                print(f"⚠️ Skipping stage {tag}")
                continue

            X = torch.load(X_path).numpy()
            Y = torch.load(Y_path).numpy()

            L0 = X.reshape(-1, cfg.dim, cfg.dim)
            D = 0.5 * (L0 + L0.transpose(0, 2, 1))

            shear_rates = [2.0 * np.sqrt(max(-compute_invariants_vectorized(d)[1], 1e-12)) for d in D]
            shear_rates = np.array(shear_rates)
            T_mats = vec6_to_sym3(Y)
            stress_norms = np.linalg.norm(T_mats, axis=(1, 2))
            cond_vals = np.array([cond(np.eye(cfg.dim) - cfg.lam * l) for l in L0])

            stage_tags.append(tag)
            shear_means.append(np.mean(shear_rates))
            stress_means.append(np.mean(stress_norms))
            cond_means.append(np.mean(cond_vals))

        os.makedirs(cfg.paths.images, exist_ok=True)
        plt.figure(figsize=(9, 6))
        plt.plot(stage_tags, shear_means, 'o-', label='Mean shear rate')
        plt.plot(stage_tags, stress_means, 's-', label='Mean ||T||')
        plt.plot(stage_tags, cond_means, '^-', label='Mean cond(A)')
        plt.legend()
        plt.title(f"Stagewise Trends ({cfg.constitutive_eq}, Stable)")
        plt.xticks(rotation=45)
        plt.yscale("log")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.paths.images, f"{cfg.constitutive_eq}_stable_trends.png"))
        plt.close()
        print(f"✅ Saved stable trends plot.")

    else:
        # Unstable mode
        X_path = os.path.join(cfg.paths.data, "unstable", f"X_{cfg.dim}D_{cfg.constitutive_eq}_unstable.pt")
        Y_path = os.path.join(cfg.paths.data, "unstable", f"Y_{cfg.dim}D_{cfg.constitutive_eq}_unstable.pt")
        if not (os.path.exists(X_path) and os.path.exists(Y_path)):
            print(f"❌ No unstable data found for {cfg.constitutive_eq}")
            return

        X = torch.load(X_path).numpy()
        Y = torch.load(Y_path).numpy()

        L0 = X.reshape(-1, cfg.dim, cfg.dim)
        T_mats = vec6_to_sym3(Y)

        

        # --- 2) Histograms: shear rate & stress norm ---
        D = 0.5 * (L0 + L0.transpose(0, 2, 1))
        shear_rates = [2.0 * np.sqrt(max(-compute_invariants_vectorized(d)[1], 1e-12)) for d in D]
        shear_rates = np.array(shear_rates)
        stress_norms = np.linalg.norm(T_mats, axis=(1, 2))

        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
        axes2[0].hist(shear_rates, bins=50, histtype='bar', color='tab:blue', alpha=0.9)
        axes2[0].set_title("Shear rate distribution (unstable)")
        axes2[0].set_xlabel("shear rate")
        axes2[0].set_ylabel("Count")

        axes2[1].hist(stress_norms, bins=50, histtype='bar', color='tab:orange', alpha=0.9)
        axes2[1].set_title("Stress norm distribution (unstable)")
        axes2[1].set_xlabel("||T||")
        axes2[1].set_ylabel("Count")

        plt.tight_layout()
        plt.savefig(os.path.join(cfg.paths.images, f"{cfg.constitutive_eq}_unstable_shear_stress_histograms.png"))
        plt.close()
        print(f"✅ Saved shear rate and stress norm histograms.")


if __name__ == "__main__":
    main()