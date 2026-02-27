import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import cond, norm
from scipy.linalg import solve_sylvester
from utils.tensors import generate_base_L_tensor  # <-- your existing function

# =========================
# Utility: Invariants
# =========================
def compute_invariants(D):
    I1 = np.trace(D)
    II = 0.5 * (I1**2 - np.trace(D @ D))
    III = np.linalg.det(D)
    return I1, II, III

# =========================
# Controlled Maxwell-B Generator
# =========================
def generate_stable_sample_maxwell_controlled(
    dim,
    eta0,
    lam,
    target_cond=None,
    stage_range=None,
    target_shear_rate=None,
    tol=0.05,
    max_T_norm=100.0
):
    """
    Controlled Maxwell-B sample generation:
    - Midpoint cond(A) targeting for stage_range mode
    - Exact cond(A) matching for target_cond mode
    - Target shear rate scaling of symmetric part D
    - T norm outlier rejection
    """
    while True:
        # Generate random L0
        L0 = generate_base_L_tensor(dim=dim, vorticity_ratio=np.random.uniform(0, 1.0))
        D = 0.5 * (L0 + L0.T)
        W = 0.5 * (L0 - L0.T)
        A = np.eye(dim) - lam * L0
        cA = cond(A)

        # Mode1: Target cond mode
        if target_cond is not None and stage_range is None:
            if abs(cA - target_cond) > tol:
                scale_factor = (cA / target_cond) ** (-1)
                L0 *= scale_factor
                D = 0.5 * (L0 + L0.T)
                W = 0.5 * (L0 - L0.T)
                A = np.eye(dim) - lam * L0
                cA = cond(A)
            if not (np.isfinite(cA) and abs(cA - target_cond) <= tol):
                continue

        # Mode2: Range mode
        elif stage_range is not None and target_cond is None:
            cond_min, cond_max = stage_range
            if cond_min <= cA <= cond_max and np.isfinite(cA):
                cond_target_mid = (cond_min + cond_max) / 2.0
                scale_factor = (cA / cond_target_mid) ** (-1)
                L0 *= scale_factor
                D = 0.5 * (L0 + L0.T)
                W = 0.5 * (L0 - L0.T)
                A = np.eye(dim) - lam * L0
                cA = cond(A)
            else:
                continue
        else:
            raise ValueError("Either target_cond OR stage_range must be provided.")

        # Target shear rate scaling
        if target_shear_rate is not None:
            _, II, _ = compute_invariants(D)
            current_shear = 2.0 * np.sqrt(max(-II, 1e-12))
            if current_shear > 0:
                # Add slight random tolerance so values aren't all identical
                variation_factor = np.random.uniform(0.98, 1.02)
                shear_scale = (target_shear_rate / current_shear) * variation_factor
                D *= shear_scale
                L0 = D + W
                A = np.eye(dim) - lam * L0
                cA = cond(A)

        # Solve Sylvester equation for Maxwell-B steady state
        B = -lam * L0.T
        C = 2.0 * eta0 * D
        try:
            T = solve_sylvester(A, B, C)
        except Exception:
            continue

        # Reject outliers
        if norm(T) > max_T_norm:
            continue

        return L0, D, W, T, cA

# =========================
# Stage definitions
# =========================
stages = [
    {"stage_tag": "1.0", "target_cond": 1.0, "stage_range": None, "target_shear_rate": 0.1},
    {"stage_tag": "1.0_1.2", "target_cond": None, "stage_range": (1.0,1.2), "target_shear_rate": 0.3},
    {"stage_tag": "1.2_1.4", "target_cond": None, "stage_range": (1.2,1.4), "target_shear_rate": 0.6},
    {"stage_tag": "1.4_1.6", "target_cond": None, "stage_range": (1.4,1.6), "target_shear_rate": 0.9},
    {"stage_tag": "1.6_1.8", "target_cond": None, "stage_range": (1.6,1.8), "target_shear_rate": 1.1},
    {"stage_tag": "1.8_2.0", "target_cond": None, "stage_range": (1.8,2.0), "target_shear_rate": 1.3},
    {"stage_tag": "2.0_2.2", "target_cond": None, "stage_range": (2.0,2.2), "target_shear_rate": 1.5},
    {"stage_tag": "2.2_2.4", "target_cond": None, "stage_range": (2.2,2.4), "target_shear_rate": 1.7},
    {"stage_tag": "2.4_2.6", "target_cond": None, "stage_range": (2.4,2.6), "target_shear_rate": 1.9},
    {"stage_tag": "2.6_2.8", "target_cond": None, "stage_range": (2.6,2.8), "target_shear_rate": 2.1},
    {"stage_tag": "2.8_3.0", "target_cond": None, "stage_range": (2.8,3.0), "target_shear_rate": 2.3}
]

# =========================
# Output folder
# =========================
out_dir = "diagnostic_plots"
os.makedirs(out_dir, exist_ok=True)

# Lists for trends
shear_means, Tnorm_means, condA_means = [], [], []

# =========================
# Main loop
# =========================
n_samples = 2000  # reduce for quick test

for stage in stages:
    tag = stage["stage_tag"]
    tcond = stage["target_cond"]
    trange = stage["stage_range"]
    tshear = stage["target_shear_rate"]

    L_vals, D_vals, T_vals, cond_vals, shear_rates = [], [], [], [], []

    for _ in range(n_samples):
        L0, D, W, T, cA = generate_stable_sample_maxwell_controlled(
            dim=3,
            eta0=1.0,
            lam=0.5,
            target_cond=tcond,
            stage_range=trange,
            target_shear_rate=tshear,
            tol=0.05,
            max_T_norm=100.0
        )
        L_vals.append(L0.flatten())
        D_vals.append(D.flatten())
        T_vals.append(T.flatten())
        cond_vals.append(cA)
        _, II, _ = compute_invariants(D)
        shear_rates.append(2.0 * np.sqrt(max(-II, 1e-12)))

    L_vals = np.array(L_vals)
    D_vals = np.array(D_vals)
    T_vals = np.array(T_vals)
    cond_vals = np.array(cond_vals)
    shear_rates = np.array(shear_rates)

    shear_means.append(np.mean(shear_rates))
    Tnorm_means.append(np.mean(np.linalg.norm(T_vals, axis=1)))
    condA_means.append(np.mean(cond_vals))

    # --- L, D, T histograms ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(L_vals.flatten(), bins=50)
    axes[0].set_title(f"L0 entries (stage {tag})")
    axes[1].hist(D_vals.flatten(), bins=50)
    axes[1].set_title(f"D entries (stage {tag})")
    axes[2].hist(T_vals.flatten(), bins=50)
    axes[2].set_title(f"T entries (stage {tag})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"hist_LDT_stage_{tag}.png"))
    plt.close()

    # --- shear rate + cond(A) histograms, safe bins ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # shear rate bins safe
    if np.max(shear_rates) - np.min(shear_rates) < 1e-12:
        bins_shear = 1
    else:
        bins_shear = 50
    axes[0].hist(shear_rates, bins=bins_shear)
    axes[0].set_title(f"Shear rate distribution (stage {tag})")

    # cond(A) bins safe
    if np.max(cond_vals) - np.min(cond_vals) < 1e-12:
        bins_cond = 1
    else:
        bins_cond = 50
    axes[1].hist(cond_vals, bins=bins_cond)
    axes[1].set_title(f"cond(A) distribution (stage {tag})")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"hist_shear_cond_stage_{tag}.png"))
    plt.close()

# =========================
# Trend plot
# =========================
fig, ax = plt.subplots(figsize=(8, 5))
stage_labels = [s["stage_tag"] for s in stages]
ax.plot(stage_labels, shear_means, marker='o', label='Mean shear rate')
ax.plot(stage_labels, Tnorm_means, marker='s', label='Mean ||T||')
ax.plot(stage_labels, condA_means, marker='^', label='Mean cond(A)')
ax.set_xticks(range(len(stages)))
ax.set_xticklabels(stage_labels, rotation=45)
ax.legend()
ax.set_title("Stage-wise trends")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "stagewise_trends.png"))
plt.close()

print(f"✅ Diagnostic plots saved in {out_dir}")