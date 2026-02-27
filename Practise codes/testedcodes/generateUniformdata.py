import numpy as np
import os
import matplotlib.pyplot as plt
from utils.tensors import generate_random_rotation_matrix

# ======================================
# Recover eigenvalues from invariants
# ======================================
def get_eigenvalues_from_invariants(neg_II, III):
    """
    Recovers eigenvalues (lambda1, lambda2, lambda3) for a trace-free tensor 
    given its invariants (-II, III).
    
    Solves the depressed cubic: lambda^3 + II*lambda - III = 0
    (Note: input is neg_II, so equation is lambda^3 - neg_II*lambda - III = 0)
    """
    # p = II (which is -neg_II)
    p = -neg_II
    q = -III

    # We use the trigonometric solution for x^3 + px + q = 0
    # Safety clamp for zero to avoid division errors
    if neg_II < 1e-9:
        return np.zeros(3)

    # R is the radius of the circle in the deviatoric plane
    R = np.sqrt(-p / 3.0)
    
    # Argument for arccos. Clamp to [-1, 1] to handle float precision noise
    acos_arg = (-q) / (2 * R**3) # derived from (-q/2) * (sqrt(-3/p)^3) roughly
    acos_arg = np.clip(acos_arg, -1.0, 1.0)
    
    theta = (1.0 / 3.0) * np.arccos(acos_arg)
    
    # The three roots
    e1 = 2 * R * np.cos(theta)
    e2 = 2 * R * np.cos(theta - 2 * np.pi / 3.0)
    e3 = 2 * R * np.cos(theta - 4 * np.pi / 3.0)
    
    return np.array([e1, e2, e3])

# =========================
# Generate D, W, L tensors
# =========================
def generate_uniform_D_field(n_samples=10000, max_neg_II=1.5, vorticity_ratio=1.0):
    """
    Generates D tensors that are uniformly distributed in the invariant space.
    """
    print(f"Generating uniform grid for approx {n_samples} samples...")
    
    # 1. Define a rectangular grid that covers the triangle
    #    Bounds: -II from 0 to max_neg_II
    #    Bounds: III from -sqrt(4/27 * max^3) to +sqrt(...)
    max_y_bound = np.sqrt((4.0/27.0) * (max_neg_II**3))
    
    # We estimate grid density to get roughly n_samples valid points
    # Area of Lumley triangle approx = (1/2) * base * height ? No, it's a cusp.
    # We'll just oversample the rectangle and filter.
    n_grid_side = int(np.sqrt(n_samples * 2.5)) 
    
    x_vals = np.linspace(0, max_neg_II, n_grid_side)
    y_vals = np.linspace(-max_y_bound, max_y_bound, n_grid_side)
    
    # Create meshgrid
    XX, YY = np.meshgrid(x_vals, y_vals)
    XX_flat = XX.flatten()
    YY_flat = YY.flatten()
    
    # 2. Filter points inside the Lumley Triangle
    #    Condition: 27 * III^2 <= 4 * (-II)^3
    #    With tolerance to avoid edge numerical issues
    discriminant = 27 * (YY_flat**2)
    limit = 4 * (XX_flat**3)
    
    mask = discriminant <= (limit + 1e-5)
    
    valid_neg_II = XX_flat[mask]
    valid_III = YY_flat[mask]
    
    # Trim to exact requested number if we have too many
    if len(valid_neg_II) > n_samples:
        indices = np.random.choice(len(valid_neg_II), n_samples, replace=False)
        valid_neg_II = valid_neg_II[indices]
        valid_III = valid_III[indices]
    
    print(f"-> Generated {len(valid_neg_II)} valid points in invariant space.")
    
    # 3. Reconstruct Tensors
    D_list, W_list, L_list = [], [], []
    
    for nII, iii in zip(valid_neg_II, valid_III):
        # A. Get Eigenvalues from invariants
        evals = get_eigenvalues_from_invariants(nII, iii)
        
        # B. Construct Diagonal Matrix
        Lambda = np.diag(evals)
        
        # C. Rotate Randomly
        R = generate_random_rotation_matrix(3)
        D = R @ Lambda @ R.T
        
        # Random antisymmetric W
        w_vec = np.random.randn(3)
        W = np.zeros((3, 3))
        W[0, 1], W[0, 2], W[1, 2] = -w_vec[2], w_vec[1], -w_vec[0]
        W = W - W.T

        # Scale W to match vorticity_ratio
        norm_D = np.linalg.norm(D)
        norm_W = np.linalg.norm(W)
        W_scaled = W * (vorticity_ratio * norm_D / norm_W) if norm_W > 1e-10 else W

        L = D + W_scaled

        D_list.append(D)
        W_list.append(W_scaled)
        L_list.append(L)
        
    return np.array(D_list), np.array(W_list), np.array(L_list)

# ==============================
# Plot invariants (-II vs III)
# =============================
def verify_and_plot(D_array):
    """
    Computes invariants back from the generated tensors and plots them
    to prove they match the target uniform distribution.
    """
    print("Verifying invariants...")
    
    # Vectorized invariant computation
    # I = trace (should be 0)
    # II = 0.5 * (I^2 - tr(D^2)) -> since I=0 -> -0.5 * tr(D^2)
    # III = det(D)
    
    # trace of D squared
    tr_D2 = np.einsum('ijk,ijk->i', D_array, D_array)
    II = -0.5 * tr_D2
    III = np.linalg.det(D_array)
    
    neg_II = -II
    
    # --- PLOTTING ---
    plt.figure(figsize=(10, 7))
    
    # 1. Draw Theoretical Boundaries
    x_b = np.linspace(0, 1.6, 1000)
    y_b = 2 * np.sqrt((x_b/3)**3) # Derived from 27*III^2 = 4*(-II)^3
    
    # Fill admissible region background
    plt.fill_between(x_b, -y_b, y_b, color='#a8c6ff', alpha=0.5, label='Admissible Region')
    plt.plot(x_b, y_b, 'k-', lw=1.5, label='Boundary')
    plt.plot(x_b, -y_b, 'k-', lw=1.5)
    plt.axhline(0, color='k', lw=0.5)

    # 2. Scatter Generated Points
    plt.scatter(neg_II, III, c='crimson', s=5, alpha=0.6, label='Generated Uniform Tensors')
    
    plt.title(f"Uniformly Distributed ({len(D_array)} Samples)", fontsize=14)
    plt.xlabel(r"$-II_D$ (Second Invariant)", fontsize=12)
    plt.ylabel(r"$III_D$ (Third Invariant)", fontsize=12)
    plt.xlim(0, 1.5)
    plt.ylim(-0.6, 0.6)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    file_path = os.path.join(output_dir, "uniform_distribution_check.png")
    plt.savefig(file_path, dpi=300)
    print(f"Plot saved to {file_path}")
    plt.close()
    
# =========================
# Main execution
# =========================
if __name__ == "__main__":
    # Main Execution
    n_samples = 10000

    # Create images/uniform folder
    output_dir = os.path.join("images", "uniform")
    os.makedirs(output_dir, exist_ok=True)

    # Generate tensors
    D_tensors, W_tensors, L_tensors = generate_uniform_D_field(n_samples=n_samples)

    # Plot invariants for D
    verify_and_plot(D_tensors)

    # --- Histogram of D tensor entries ---
    D_flat = D_tensors.flatten()  # flatten to 1D array
    plt.figure(figsize=(8, 5))
    plt.hist(D_flat, bins=50, color='steelblue', edgecolor='black')
    plt.xlabel("D tensor values", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Histogram of D tensor ", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    file_path = os.path.join(output_dir, "D_tensor_histogram.png")
    plt.savefig(file_path, dpi=300)
    print(f"Histogram of D tensor entries saved to {file_path}")
    
    # Histogram: L entries
    plt.figure(figsize=(8, 5))
    plt.hist(L_tensors.flatten(), bins=50, color='darkorange', edgecolor='black')
    plt.xlabel("L tensor values")
    plt.ylabel("Frequency")
    plt.title("Histogram of L tensor ")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    file_path = os.path.join(output_dir, "L_tensor_histogram.png")
    plt.savefig(file_path, dpi=300)
    print(f"Histogram of L tensor entries saved to {file_path}")
    