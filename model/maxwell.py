import numpy as np
from numpy.linalg import cond
from scipy.linalg import solve_sylvester
from utils.solver_projection import solve_symmetric_sylvester_kron  # NEW import

def solve_steady_state_maxwell(L, eta0=5.28e-5, lam=1.902,
                               return_cond=False, debug=False, sample_idx=None,
                               use_projection=False):
    """
    Solve steady-state Maxwell-B with either SciPy or projection solver.
    """
    L = L.astype(np.float64)
    dim = L.shape[0]
    D = 0.5 * (L + L.T)

    # Setup (plus form for SciPy/projection)
    A = (1 / lam) * np.eye(dim) - L.T
    B = -L
    C = (2 * eta0 / lam) * D

    if use_projection:
        T = solve_symmetric_sylvester_kron(A, B, C)
        condM = cond(A)  # we can also compute cond of reduced system if needed
    else:
        T = solve_sylvester(A, B, C)
        condM = cond(A)

    R = A @ T + T @ B - C
    resid = np.linalg.norm(R, 'fro')
    sym_err = np.linalg.norm(T - T.T, 'fro')

    if debug:
        solver_name = "Projection" if use_projection else "SciPy"
        print(f"[Maxwell-B-{solver_name}] Sample {sample_idx} Residual={resid:.3e}, SymErr={sym_err:.3e}")

    if return_cond:
        return T, condM, resid
    else:
        return T