import numpy as np

def get_eigenvalues_from_invariants(neg_II, III):
    """
    Recovers eigenvalues (lambda1, lambda2, lambda3) for a trace-free tensor 
    given its invariants (-II, III).
    
    Solves the depressed cubic equation: lambda^3 - neg_II*lambda - III = 0
    
    Parameters:
    -----------
    neg_II : float
        The negative of the second invariant (-II_D). Must be >= 0.
    III : float
        The third invariant (determinant).
        
    Returns:
    --------
    np.array
        The three eigenvalues [e1, e2, e3].
    """
    p = -neg_II
    q = -III

    # Safety: neg_II must be non-negative for real roots in this physics context
    if neg_II < 1e-9:
        return np.zeros(3)

    # R is the radius of the circle in the deviatoric plane
    R = np.sqrt(-p / 3.0)
    
    # Calculate argument for arccos. 
    # We clip to [-1, 1] to prevent NaN errors from tiny floating point noise.
    acos_arg = (-q) / (2 * R**3)
    acos_arg = np.clip(acos_arg, -1.0, 1.0)
    
    # Trigonometric solution for cubic roots
    theta = (1.0 / 3.0) * np.arccos(acos_arg)
    
    e1 = 2 * R * np.cos(theta)
    e2 = 2 * R * np.cos(theta - 2 * np.pi / 3.0)
    e3 = 2 * R * np.cos(theta - 4 * np.pi / 3.0)
    
    return np.array([e1, e2, e3])


def generate_invariant_grid(n_samples_needed, max_neg_II=1.5, oversample_factor=1.0):
    """
    Generates a structured GRID of points uniformly distributed inside the Lumley Triangle.
    
    LOGIC UPDATE:
    - oversample_factor=1.0: We strictly generate 'n_samples_needed' points for the 
      bounding box (e.g., 10,000 raw points).
    - No Shuffle: Points are processed in exact grid order. This ensures the final 
      plots look like a solid object with 'holes' (where physics rejected points) 
      rather than scattered random noise.
    
    Parameters:
    -----------
    n_samples_needed : int
        The raw number of grid points for the bounding box.
    max_neg_II : float
        The maximum extent of the second invariant (x-axis limit).
    oversample_factor : float
        Multiplier for grid size. Set to 1.0 for a strict single-pass grid.
        
    Returns:
    --------
    np.ndarray
        Array of shape (N, 2) containing valid [neg_II, III] pairs inside the triangle.
    """
    print(f"   -> Generating fixed grid candidates (Factor: {oversample_factor}x)...")

    # 1. Exact raw count (e.g., 10,000)
    total_candidates = int(n_samples_needed * oversample_factor)
    
    # 2. Grid dimensions (e.g., 100x100 for 10k points)
    grid_side = int(np.sqrt(total_candidates)) 
    
    # 3. Define Bounding Box Dimensions
    # The Lumley triangle is bounded by y = +/- 2/sqrt(27) * x^(3/2)
    max_y_bound = np.sqrt((4.0/27.0) * (max_neg_II**3))
    
    x_vals = np.linspace(0, max_neg_II, grid_side)
    y_vals = np.linspace(-max_y_bound, max_y_bound, grid_side)
    
    # 4. Create 2D Meshgrid
    XX, YY = np.meshgrid(x_vals, y_vals)
    XX_flat = XX.flatten()
    YY_flat = YY.flatten()
    
    # 5. Filter for Lumley Triangle Logic
    # Mathematical Condition: 27 * III^2 <= 4 * (-II)^3
    discriminant = 27 * (YY_flat**2)
    limit = 4 * (XX_flat**3)
    
    # We verify mask with a tiny tolerance (1e-5) to include boundary points
    mask = discriminant <= (limit + 1e-5)
    
    valid_neg_II = XX_flat[mask]
    valid_III = YY_flat[mask]
    
    # 6. Stack into (N, 2) array
    grid_points = np.stack([valid_neg_II, valid_III], axis=1)
    
    # NO SHUFFLE: Keeps the perfect visual grid structure.
    
    print(f"   -> Grid generated. {len(grid_points)} candidates inside triangle (from {total_candidates} raw grid points).")
    return grid_points