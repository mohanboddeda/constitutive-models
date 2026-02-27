import numpy as np

def compute_invariants_vectorized(D):
    if D.ndim == 2: 
        D = D[np.newaxis, :, :]
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
    import numpy as np
    # Reuse your invariants function
    D0_list = [0.5 * (L + L.T) for L in L0_list]
    D0_array = np.array(D0_list)

    I, II, III = compute_invariants_vectorized(D0_array)
    discriminant = ((-III / 2)**2 + (-II / 3)**3)
    tolerance = 1e-1

    kept_mask = discriminant <= tolerance
    filtered_L0_list = [L for L, keep in zip(L0_list, kept_mask) if keep]

    return filtered_L0_list, kept_mask