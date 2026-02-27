import numpy as np

def compute_invariants_vectorized(D):
    if D.ndim == 2: 
        D = D[np.newaxis, :, :]
    I = np.trace(D, axis1=-2, axis2=-1)
    II = 0.5 * (I**2 - np.trace(D @ D, axis1=-2, axis2=-1))
    III = np.linalg.det(D) if D.shape[-1] == 3 else np.zeros(D.shape[0])
    return I, II, III