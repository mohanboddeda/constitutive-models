import random
import numpy as np
from utils.invariants import compute_invariants_vectorized

def carreau_yasuda_residual_check(L0_list, Y_list,
                                  nu_0=5.28e-5,
                                  nu_inf=3.30e-6,
                                  lambda_val=1.902,
                                  n=0.22,
                                  a=1.25,
                                  n_samples=3):
    """
    Picks a few random samples from Carreau–Yasuda dataset and checks whether
    ν_data matches ν_model(L) at machine precision.

    Parameters
    ----------
    L0_list : list or array
        List of L tensors from dataset generation.
    Y_list : list or array
        List of viscosities computed in dataset generation.
    nu_0, nu_inf, lambda_val, n, a : float
        Carreau–Yasuda parameters (must match generation settings).
    n_samples : int
        Number of random samples to check.
    """
    random_indices = random.sample(range(len(L0_list)), n_samples)
    print("\n[Carreau–Yasuda Residual Check]")
    print(f"{'Sample':>8} {'ν_data (Pa·s)':>18} {'ν_model (Pa·s)':>18} {'Residual (Pa·s)':>18}")

    for idx in random_indices:
        L = L0_list[idx]
        nu_data = Y_list[idx]

        # Compute shear rate from L
        D = 0.5 * (L + L.T)
        _, second_invariant_D, _ = compute_invariants_vectorized(D)
        second_invariant_D = -second_invariant_D
        gamma_dot = 2 * np.sqrt(second_invariant_D + 1e-12)[0]

        # Compute viscosity from Carreau–Yasuda formula
        nu_model = nu_inf + (nu_0 - nu_inf) * (1 + (lambda_val * gamma_dot)**a)**((n - 1) / a)

        # Residual
        residual = nu_data - nu_model
        print(f"{idx:8d} {nu_data:18.6e} {nu_model:18.6e} {residual:18.3e}")