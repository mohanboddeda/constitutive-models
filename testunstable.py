import numpy as np
from numpy.linalg import cond
from scipy.linalg import solve_sylvester
import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf
import os

from utils.tensors import generate_base_L_tensor
from utils.plotting import plot_invariants_diagram

@hydra.main(config_path="config/data", config_name="dataConfig", version_base=None)
def main(cfg):
    np.random.seed(cfg.seed)

    print(f"=== Checking cond(A) range for {cfg.n_samples} unstable samples ===")
    conds = []
    L_list = []
    T_list = []

    for i in range(cfg.n_samples):
        v_ratio = np.random.uniform(0, 5) 
        L0 = generate_base_L_tensor(dim=cfg.dim, vorticity_ratio=v_ratio)
        lamda = 0.2
        eta0 = 5.0

        D = 0.5 * (L0 + L0.T)
        A = np.eye(cfg.dim) - lamda * L0
        B = -lamda * L0.T
        C = 2.0 * eta0 * D

        cA = cond(A)
        T = solve_sylvester(A, B, C)

        if np.isfinite(cA):
            conds.append(cA)
            L_list.append(L0)
            T_list.append(T)

    conds = np.array(conds)
    print(f"Min cond(A):    {conds.min():.6f}")
    print(f"Max cond(A):    {conds.max():.6f}")
    print(f"Median cond(A): {np.median(conds):.6f}")
    print(f"Mean cond(A):   {np.mean(conds):.6f}")

    # Histogram of cond(A)
    plt.figure(figsize=(6,4))
    plt.hist(conds, bins=200, range=(1,50))
    plt.xlabel('cond(A)')
    plt.ylabel('Frequency')
    plt.title('Histogram of cond(A) for unstable case')
    plt.show()

    # === Side-by-side histograms for L and T ===
    L_flat = np.array(L_list).flatten()
    T_flat = np.array(T_list).flatten()

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.hist(L_flat, bins=50, color='steelblue', edgecolor='white')
    plt.title("Distribution of L tensor elements")
    plt.xlabel("Value")
    plt.ylabel("Count")

    plt.subplot(1,2,2)
    plt.hist(T_flat, bins=50, color='orange', edgecolor='white')
    plt.title("Distribution of T tensor elements")
    plt.xlabel("Value")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()

    # === Call invariants diagram plot ===
    print("Plotting invariant diagram for sampled L tensors...")
    plot_invariants_diagram(cfg, L_list, title="Invariant Diagram for Unstable Samples")

if __name__ == "__main__":
    main()