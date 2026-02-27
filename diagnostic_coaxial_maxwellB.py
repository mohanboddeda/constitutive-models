import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_sylvester
from numpy.linalg import cond

# ============================================================
# 1. Rotation & Tensor Utilities
# ============================================================

def generate_random_rotation_matrix(dim=3):
    H = np.random.randn(dim, dim)
    Q, _ = np.linalg.qr(H)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def generate_coaxial_L(dim, R_fixed, vorticity_ratio=1.0):
    # --- eigenvalues only ---
    ew = np.random.randn(dim)
    ew -= np.mean(ew)

    D = R_fixed @ np.diag(ew) @ R_fixed.T

    # --- antisymmetric W ---
    w = np.random.randn(dim)
    W = np.zeros((dim, dim))
    W[0, 1], W[0, 2], W[1, 2] = -w[2], w[1], -w[0]
    W = W - W.T

    if np.linalg.norm(W) > 1e-12:
        W *= vorticity_ratio * np.linalg.norm(D) / np.linalg.norm(W)

    return D + W, D


# ============================================================
# 2. Maxwell‑B Solver
# ============================================================

def solve_maxwellB(L, eta0, lam):
    D = 0.5 * (L + L.T)
    A = np.eye(3) - lam * L
    B = -lam * L.T
    C = 2.0 * eta0 * D
    T = solve_sylvester(A, B, C)
    return 0.5 * (T + T.T)


# ============================================================
# 3. Condition‑controlled sample generation
# ============================================================

def generate_stable_coaxial_sample(
    R_fixed,
    eta0,
    lam,
    target_cond=2.5,
    tol=0.05,
    vorticity_ratio=1.0,
    max_tries=200
):
    """
    Generate ONE coaxial L with controlled cond(I - λL)
    """
    for _ in range(max_tries):
        L, D = generate_coaxial_L(
            dim=3,
            R_fixed=R_fixed,
            vorticity_ratio=vorticity_ratio
        )

        A = np.eye(3) - lam * L
        cA = cond(A)

        if not np.isfinite(cA):
            continue

        # --- scale L to hit target condition ---
        scale = (cA / target_cond) ** (-1)
        L *= scale
        D *= scale

        A = np.eye(3) - lam * L
        cA = cond(A)

        if abs(cA - target_cond) > tol:
            continue

        try:
            T = solve_maxwellB(L, eta0, lam)

            # Residual check
            R = A @ T + T @ (-lam * L.T) - 2 * eta0 * D
            if np.linalg.norm(R, "fro") < 1e-10:
                return L, D, T, cA

        except Exception:
            continue

    raise RuntimeError("Failed to generate stable sample")


# ============================================================
# 4. Invariants
# ============================================================

def invariants(D):
    I = np.trace(D)
    II = 0.5 * (I**2 - np.trace(D @ D))
    III = np.linalg.det(D)
    return -II, III


# ============================================================
# 5. Main Diagnostics
# ============================================================

def run_diagnostics(
    n_samples=20000,
    eta0=1.0,
    lam=0.6,
    target_cond=2.6,
    vorticity_ratio=0.5,
    seed=42
):
    np.random.seed(seed)

    R_FIXED = generate_random_rotation_matrix()

    L_list, T_list, inv_list = [], [], []

    for _ in range(n_samples):
        L, D, T, cA = generate_stable_coaxial_sample(
            R_FIXED,
            eta0,
            lam,
            target_cond=target_cond,
            vorticity_ratio=vorticity_ratio
        )

        L_list.append(L)
        T_list.append(T)
        inv_list.append(invariants(D))

    L = np.array(L_list)
    T = np.array(T_list)
    inv = np.array(inv_list)

    print(f"✅ Generated {len(L)} stable coaxial samples")
    print(f"✅ Mean cond(I-λL): {np.mean([cond(np.eye(3)-lam*l) for l in L]):.3f}")

    # ========================================================
    # Lumley Triangle
    # ========================================================

    plt.figure(figsize=(7, 6))
    plt.scatter(inv[:, 0], inv[:, 1], s=10, alpha=0.6)

    x = np.linspace(0, 2.0, 400)
    y = 2 * np.sqrt((x / 3) ** 3)
    plt.plot(x, y, "k")
    plt.plot(x, -y, "k")
    plt.axhline(0, color="k")

    plt.xlabel(r"$-II_D$")
    plt.ylabel(r"$III_D$")
    plt.title("Invariant Diagram (Coaxial, Condition‑Controlled)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ========================================================
    # Histograms
    # ========================================================

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    axes[0].hist(L.flatten(), bins=80, log=True)
    axes[0].set_title("Velocity Gradient $L$")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(T.flatten(), bins=80, log=True)
    axes[1].set_title("Stress Tensor $T$")
    axes[1].set_xlabel("Value")

    plt.tight_layout()
    plt.show()


# ============================================================
# 6. Run
# ============================================================

if __name__ == "__main__":
    run_diagnostics(
        n_samples=3000,
        eta0=1.0,
        lam=0.6,
        target_cond=1.6,
        vorticity_ratio=0.5
    )