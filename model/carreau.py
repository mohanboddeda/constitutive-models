import numpy as np
from utils.invariants import compute_invariants_vectorized

def carreau_yasuda_viscosity(L, nu_0, nu_inf, lambda_val, n, a):
    """
    Compute steady-state viscosity from Carreau-Yasuda model.
    """
    D = 0.5 * (L + L.T)
    _, second_invariant_D, _ = compute_invariants_vectorized(D)
    second_invariant_D = -second_invariant_D
    epsilon = 1e-12
    shear_rate = 2 * np.sqrt(second_invariant_D + epsilon)[0]
    term1 = (lambda_val * shear_rate) ** a
    term2 = (1 + term1) ** ((n - 1) / a)
    nu = nu_inf + (nu_0 - nu_inf) * term2
    return nu