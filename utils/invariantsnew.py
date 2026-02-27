import numpy as np

def compute_invariants_vectorized(D):
    """
    Computes the three invariants for a batch of tensors D.
    Shape D: (N, 3, 3) or (3, 3)
    """
    # Handle single matrix case
    if D.ndim == 2: 
        D = D[np.newaxis, :, :]
        
    # Handle empty array case (prevent crash if N=0)
    if D.size == 0:
        return np.array([]), np.array([]), np.array([])

    I = np.trace(D, axis1=-2, axis2=-1)
    II = 0.5 * (I**2 - np.trace(D @ D, axis1=-2, axis2=-1))
    III = np.linalg.det(D) if D.shape[-1] == 3 else np.zeros(D.shape[0])
    return I, II, III

def filter_admissible_region(L0_list):
    """
    Filters out samples whose D invariants are outside theoretical admissible region.

    Parameters:
        L0_list (list of np.ndarray): velocity gradient tensors L0 for each sample.

    Returns:
        filtered_L0_list: list of accepted L0 tensors
        kept_mask: boolean mask showing which samples are kept
    """
    # --- SAFETY CHECK: Handle Empty Inputs ---
    if not L0_list:
        return [], []

    # Calculate symmetric part D
    D0_list = [0.5 * (L + L.T) for L in L0_list]
    D0_array = np.array(D0_list)

    # Compute Invariants
    I, II, III = compute_invariants_vectorized(D0_array)
    
    # Lumley Triangle Condition: 27*III^2 <= 4*(-II)^3
    # Note: II is negative for trace-free tensors, so (-II) is positive.
    discriminant = ((-III / 2)**2 + (-II / 3)**3)
    
    # Tolerance for numerical noise (1e-1 is loose, but standard for this check)
    tolerance = 1e-1

    kept_mask = discriminant <= tolerance
    
    # Filter the list
    filtered_L0_list = [L for L, keep in zip(L0_list, kept_mask) if keep]

    return filtered_L0_list, kept_mask