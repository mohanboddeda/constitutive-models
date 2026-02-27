import numpy as np
import matplotlib.pyplot as plt
import os
np.random.seed(42)
from utils.tensors import generate_random_rotation_matrix 

def generate_base_tensor(eigenvalues):
    """Given eigenvalues, return symmetric tensor D in random orientation."""
    R = generate_random_rotation_matrix(len(eigenvalues))
    Lambda = np.diag(eigenvalues)
    return R @ Lambda @ R.T

def compute_invariants(D):
    """Return invariants I, II, III for symmetric matrix D."""
    I = np.trace(D)
    II = 0.5 * (I**2 - np.trace(D @ D))
    III = np.linalg.det(D)
    return I, II, III

def get_flow_eigenvalues(flow_type, rate=1.0):
    """Return canonical eigenvalues for each flow type."""
    if flow_type == "uniaxial_extension":
        return [rate, -rate/2, -rate/2]
    elif flow_type == "biaxial_extension":
        return [rate, rate, -2*rate]
    elif flow_type == "planar_extension":
        return [rate, -rate, 0]
    elif flow_type == "pure_shear":
        return [rate/2, -rate/2, 0]
    elif flow_type == "mixed_flow_above":
        return [rate, -rate/4, -3*rate/4]
    elif flow_type == "mixed_flow_below":
        return [rate, 0.6*rate, -1.6*rate]
    else:
        raise ValueError("Unknown flow type")

def generate_random_W_tensor(D0, vorticity_ratio=1.0):
    """Generate a random antisymmetric W tensor, scaled to D norm."""
    dim = D0.shape[0]
    w = np.random.randn(dim)  # random vorticity vector
    W0 = np.zeros((dim, dim))
    
    if dim == 3:
        # Fill upper triangle from w-vector then antisymmetrize
        W0[0, 1], W0[0, 2], W0[1, 2] = -w[2], w[1], -w[0]
        W0 = W0 - W0.T
    elif dim == 2:
        W0[0, 1] = -w[0]
        W0[1, 0] = w[0]
    
    norm_D0 = np.linalg.norm(D0)
    norm_W0 = np.linalg.norm(W0)
    
    if norm_W0 > 1e-10:
        W0 *= (vorticity_ratio * norm_D0 / norm_W0)
    
    return W0

# =========================
# Flow types configuration
# =========================
flow_types = [
    "uniaxial_extension",
    "biaxial_extension",
    "planar_extension",
    "pure_shear",
    "mixed_flow_above",
    "mixed_flow_below"
]  
colors = {
    "uniaxial_extension": "blue",
    "biaxial_extension": "green",
    "planar_extension": "orange",
    "pure_shear": "red",
    "mixed_flow_above": "purple",
    "mixed_flow_below": "brown"
}

labels = {
    "uniaxial_extension": "Uniaxial ext.",
    "biaxial_extension": "Biaxial ext.",
    "planar_extension": "Planar ext.",
    "pure_shear": "Pure shear",
    "mixed_flow_above": "Mixed above axis",
    "mixed_flow_below": "Mixed below axis"
}

rate_min, rate_max = 0.0, 2.0
N_samples = 100000
points = {ft: [] for ft in flow_types}

# Ensure base output folder exists
base_folder = "flow_data"
os.makedirs(base_folder, exist_ok=True)

for ft in flow_types:
    print(f"\n=== Flow type: {ft} ===")
    canonical_ev = get_flow_eigenvalues(ft, 1.0)
    print("Canonical eigenvalues ratios (rate=1):", canonical_ev)

    folder_path = os.path.join(base_folder, ft)
    os.makedirs(folder_path, exist_ok=True)
    txt_file_path = os.path.join(folder_path, f"{ft}_samples.txt")

    with open(txt_file_path, "w") as ftxt:
        ftxt.write(f"=== Flow type: {ft} ===\n")
        ftxt.write(f"Canonical eigenvalues ratios (rate=1): {canonical_ev}\n")

        D_list, W_list, L_list, invariants_list = [], [], [], []

        for i in range(N_samples):
            rate = np.random.uniform(rate_min, rate_max)
            ev = get_flow_eigenvalues(ft, rate)
            D = generate_base_tensor(ev)
            W = generate_random_W_tensor(D, vorticity_ratio=1.0)
            L = D + W

            I, II, III = compute_invariants(D)
            points[ft].append((-II, III))

            D_list.append(D)
            W_list.append(W)
            L_list.append(L)
            invariants_list.append([I, II, III])

            if i < 3:
                eigvals_unsorted = np.linalg.eigvals(D)
                L_norm = np.linalg.norm(L, 'fro')
                D_norm = np.linalg.norm(D, 'fro')
                W_norm = np.linalg.norm(W, 'fro')

                ftxt.write(f"\nSample {i+1}:\n")
                ftxt.write(f"Rate: {rate:.3f}\n")
                ftxt.write(f"Canonical eigenvalues: {np.round(ev, 3)}\n")
                ftxt.write(f"D matrix (rotated):\n{np.round(D, 3)}\n")
                ftxt.write(f"Eigenvalues of D(rotated): {np.round(eigvals_unsorted, 3)}\n")
                ftxt.write(f"L matrix:\n{np.round(L, 3)}\n")
                ftxt.write(f"W matrix:\n{np.round(W, 3)}\n")
                ftxt.write(f"L norm = {L_norm:.4e}\n")
                ftxt.write(f"D norm = {D_norm:.4e}\n")
                ftxt.write(f"W norm = {W_norm:.4e}\n")

        # Save arrays
        np.save(os.path.join(folder_path, "D.npy"), np.array(D_list))
        np.save(os.path.join(folder_path, "W.npy"), np.array(W_list))
        np.save(os.path.join(folder_path, "L.npy"), np.array(L_list))
        np.save(os.path.join(folder_path, "invariants.npy"), np.array(invariants_list))

print("Data and sample info saved in 'flow_data' folder.")

# Plot invariant diagram
plt.figure(figsize=(8, 6))
II_vals = np.linspace(0, 1.5, 200)
III_max = (2 / (3 * np.sqrt(3))) * (II_vals ** 1.5)
plt.fill_between(II_vals, III_max, -III_max, color="lightblue", alpha=0.5)
plt.plot(II_vals, III_max, 'k-', linewidth=2)
plt.plot(II_vals, -III_max, 'k-', linewidth=2)

for ft in flow_types:
    pts = points[ft]
    plt.scatter([p[0] for p in pts], [p[1] for p in pts],
                color=colors[ft], alpha=0.7, label=labels[ft])

plt.xlabel(r"Second Invariant ($-II_{\mathbf{D}}$)", fontsize=14)
plt.ylabel(r"Third Invariant ($III_{\mathbf{D}}$)", fontsize=14)
plt.title("Invariant Diagram for different Flow Types", fontsize=16, pad=10)
plt.grid(True)
plt.legend()
plt.xlim(0, 1.5)
plt.ylim(-0.5, 0.5)
plt.tight_layout()

plot_path = os.path.join("flow_data", "Invariant_Diagram.png")
plt.savefig(plot_path, dpi=300)
print(f"Invariant diagram saved as {plot_path}")

