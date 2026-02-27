import numpy as np
from numpy.linalg import cond, norm
from scipy.linalg import solve_sylvester
import matplotlib.pyplot as plt

# ===== Helper functions =====
def generate_random_rotation_matrix(dim=3):
    H = np.random.randn(dim, dim)
    Q, _ = np.linalg.qr(H)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q

def generate_base_tensor(eigenvalues):
    dim = len(eigenvalues)
    eigenvalues = np.random.permutation(eigenvalues)
    R = generate_random_rotation_matrix(dim)
    Lambda = np.diag(eigenvalues)
    return R @ Lambda @ R.T

def generate_base_L_tensor(dim=3, vorticity_ratio=1.0):
    ew = np.random.randn(dim)
    ew -= np.mean(ew)
    D0 = generate_base_tensor(ew)

    w = np.random.randn(dim)
    W0 = np.zeros((dim, dim))
    if dim == 3:
        W0[0, 1], W0[0, 2], W0[1, 2] = -w[2], w[1], -w[0]
        W0 = W0 - W0.T
    elif dim == 2:
        W0[0, 1] = -w[0]
        W0[1, 0] = w[0]

    norm_D0 = np.linalg.norm(D0)
    norm_W0 = np.linalg.norm(W0)
    if norm_W0 > 1e-10:
        W0_scaled = W0 * (vorticity_ratio * norm_D0 / norm_W0)
    else:
        W0_scaled = W0
    return D0 + W0_scaled

def flatten_symmetric(tensors):
    dim = tensors.shape[-1]
    if dim == 3:
        return np.stack([
            tensors[..., 0, 0], tensors[..., 1, 1], tensors[..., 2, 2],
            tensors[..., 0, 1], tensors[..., 0, 2], tensors[..., 1, 2]
        ], axis=-1)
    elif dim == 2:
        return np.stack([
            tensors[..., 0, 0], tensors[..., 1, 1], tensors[..., 0, 1]
        ], axis=-1)

# ===== Main sampler =====
def sample_variable_cond_data(n_samples=50000, dim=3, eta0=1.0, lam=0.5):
    L_list, D_list, T_list, condA_list = [], [], [], []

    for i in range(n_samples):
        v_ratio = np.random.uniform(0, 1.0)
        L0 = generate_base_L_tensor(dim=dim, vorticity_ratio=v_ratio)

        target_cond = np.random.uniform(1.5, 2.0) 
        A = np.eye(dim) - lam * L0
        cA = cond(A)

        if abs(cA - target_cond) > 0.05:
            scale_factor = (cA / target_cond) ** (-1)
            L0 *= scale_factor
            A = np.eye(dim) - lam * L0
            cA = cond(A)

        D = 0.5 * (L0 + L0.T) 
        B = -lam * L0.T
        C = 2.0 * eta0 * D
        T = solve_sylvester(A, B, C)

        L_list.append(L0)
        D_list.append(D)
        T_list.append(T)
        condA_list.append(cA)

        if i < 3:
            print("=" * 40)
            print(f"Sample {i} (cond(A)={cA:.4f})")
            print("L:\n", np.array_str(L0, precision=4, suppress_small=True))
            print("D:\n", np.array_str(D, precision=4, suppress_small=True))
            print("T:\n", np.array_str(T, precision=4, suppress_small=True))
            print(f"L norm={norm(L0):.4f}, D norm={norm(D):.4f}, T norm={norm(T):.4f}")
            print(f"T symmetry error={norm(T - T.T):.2e}")

    # Flatten arrays
    X_flat = np.array(L_list).reshape(n_samples, -1)
    Y_flat = flatten_symmetric(np.array(T_list))

    print("\n=== X column min/max ===")
    print("Min:", np.min(X_flat, axis=0))
    print("Max:", np.max(X_flat, axis=0))
    print("\n=== Y column min/max ===")
    print("Min:", np.min(Y_flat, axis=0))
    print("Max:", np.max(Y_flat, axis=0))

    # ===== Outlier detection based on T norm =====
    T_norms = np.linalg.norm(np.array(T_list), axis=(1, 2))
    median_norm = np.median(T_norms)
    iqr = np.percentile(T_norms, 75) - np.percentile(T_norms, 25)
    k = 3
    threshold = median_norm + k * iqr

    outlier_indices = np.where(T_norms > threshold)[0]

    print("\n=== Outlier detection ===")
    print(f"Median T norm: {median_norm:.4f}, IQR: {iqr:.4f}, Threshold: {threshold:.4f}")
    print(f"Outlier count: {len(outlier_indices)} / {n_samples}")
    if len(outlier_indices) > 0:
        print(f"Max T norm in outliers: {T_norms[outlier_indices].max():.4f}")

    # Remove outliers from dataset
    X_clean = np.delete(X_flat, outlier_indices, axis=0)
    Y_clean = np.delete(Y_flat, outlier_indices, axis=0)

    print(f"Clean dataset size: {X_clean.shape[0]} samples (removed {len(outlier_indices)})")

    return X_clean, Y_clean, outlier_indices, T_norms

# ===== Run sampler =====
if __name__ == "__main__":
    X_clean, Y_clean, outliers, T_norms = sample_variable_cond_data(n_samples=50000, dim=3, eta0=1.0, lam=0.5)