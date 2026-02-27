import numpy as np

def compute_invariants(L):
    """
    Compute a basic set of rotational invariants from a velocity gradient L (3x3).
    Returns invariants as 1D numpy array.
    """
    D = 0.5 * (L + L.T)  # symmetric
    W = 0.5 * (L - L.T)  # antisymmetric

    I1 = np.trace(D)
    I2 = 0.5 * (np.trace(D)**2 - np.trace(D @ D))
    I3 = np.linalg.det(D)

    J2 = -0.5 * np.trace(W @ W)            # second invariant of W, sign‐adjusted
    # Example mixed invariant:
    K1 = np.trace(D @ (W @ W))

    return np.array([I1, I2, I3, J2, K1])