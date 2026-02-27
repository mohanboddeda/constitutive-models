import os
import numpy as np
import matplotlib.pyplot as plt
from utils.data_utils_maxwellB import load_and_normalize_data_maxwellB

# Helper to convert 6-vector to full 3x3 symmetric tensor
def vec6_to_sym3_numpy(vec6):
    """
    Convert N×6 vector form [xx, yy, zz, xy, xz, yz] to N×3×3 symmetric matrix.
    """
    N = vec6.shape[0]
    T = np.zeros((N, 3, 3))
    T[:, 0, 0] = vec6[:, 0]
    T[:, 1, 1] = vec6[:, 1]
    T[:, 2, 2] = vec6[:, 2]
    T[:, 0, 1] = T[:, 1, 0] = vec6[:, 3]
    T[:, 0, 2] = T[:, 2, 0] = vec6[:, 4]
    T[:, 1, 2] = T[:, 2, 1] = vec6[:, 5]
    return T

# Function to print stats and return Frobenius norms
def describe_Y(name, Y_set):
    print(f"\n{name} Y stats (physical units):")
    Y_arr = np.array(Y_set)   # shape (N,6)
    print("Min per component:", np.min(Y_arr, axis=0))
    print("Max per component:", np.max(Y_arr, axis=0))
    print("Std per component:", np.std(Y_arr, axis=0))
    # Convert to full tensor
    Y_full = vec6_to_sym3_numpy(Y_arr)
    frob = np.linalg.norm(Y_full, 'fro', axis=(1, 2))
    print(f"Frobenius norm avg = {frob.mean():.3f}")
    print(f"Frobenius norm std = {frob.std():.3f}")
    print(f"Frobenius norm min = {frob.min():.3f}")
    print(f"Frobenius norm max = {frob.max():.3f}")
    return frob, Y_arr

# ---------- Main Script ----------
X_path = "./datafiles/X_3D_maxwell_B.pt"
Y_path = "./datafiles/Y_3D_maxwell_B.pt"

# Load normalized data (minmax or standard scaling doesn't affect physical stats prints here)
X_train_hat, X_val_hat, X_test_hat, \
Y_train_hat, Y_val_hat, Y_test_hat, \
X_mean, X_std, Y_mean, Y_std = load_and_normalize_data_maxwellB(
    X_path, Y_path,
    seed=42,
    test_size=0.1,
    val_size=0.2,
    balanced_split=True,
    scaling_mode="standard"  # change to "minmax" if needed
)

# Output directory for plots
fig_dir = "./data_distribution_plots"
os.makedirs(fig_dir, exist_ok=True)

# Dataset sizes
print("\n=== Dataset sizes ===")
print(f"Train: {X_train_hat.shape[0]} samples")
print(f"Val:   {X_val_hat.shape[0]} samples")
print(f"Test:  {X_test_hat.shape[0]} samples")

# Stats
frob_train, comp_train = describe_Y("Train", Y_train_hat)
frob_val,   comp_val   = describe_Y("Val",   Y_val_hat)
frob_test,  comp_test  = describe_Y("Test",  Y_test_hat)

# ----------- Histograms for Frobenius Norms --------------
plt.figure(figsize=(8, 5))
plt.hist(frob_train, bins=30, alpha=0.5, label="Train", color='tab:blue', density=True)
plt.hist(frob_val, bins=30, alpha=0.5, label="Val", color='tab:orange', density=True)
plt.hist(frob_test, bins=30, alpha=0.5, label="Test", color='tab:green', density=True)
plt.xlabel("Frobenius Norm of Stress Tensor", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.title("Stress Magnitude Distribution — Train / Val / Test", fontsize=14)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "frob_norm_distribution.png"), dpi=300)
plt.close()
print(f"✅ Saved Frobenius norm distribution plot to {fig_dir}/frob_norm_distribution.png")

# ----------- Histograms for Each Component --------------
component_names = ["T_xx", "T_yy", "T_zz", "T_xy", "T_xz", "T_yz"]

for i, comp in enumerate(component_names):
    plt.figure(figsize=(8, 5))
    plt.hist(comp_train[:, i], bins=30, alpha=0.5, label="Train", color='tab:blue', density=True)
    plt.hist(comp_val[:, i], bins=30, alpha=0.5, label="Val", color='tab:orange', density=True)
    plt.hist(comp_test[:, i], bins=30, alpha=0.5, label="Test", color='tab:green', density=True)
    plt.xlabel(f"{comp} value (Pa)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title(f"Distribution of {comp} — Train / Val / Test", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    filename = f"{comp}_distribution.png"
    plt.savefig(os.path.join(fig_dir, filename), dpi=300)
    plt.close()
    print(f"✅ Saved {comp} distribution plot to {fig_dir}/{filename}")

print("\n📊 All distribution plots saved in:", fig_dir)