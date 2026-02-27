import numpy as np
from numpy.linalg import cond
from scipy.linalg import solve_sylvester
from utils.solver_projection import solve_symmetric_sylvester_kron  # NEW import

def solve_steady_state_oldroyd(L, eta0, lam, lam_r,
                               return_cond=False, debug=False, sample_idx=None,
                               use_projection=False):
    """
    Solve steady-state Oldroyd-B with optional projection solver.
    """
    L = L.astype(np.float64)
    dim = L.shape[0]
    D = 0.5 * (L + L.T)

    A = np.eye(dim) - lam * L
    B = -(lam * L.T)
    C = 2 * eta0 * (D - lam_r * (L @ D) - lam_r * (D @ L.T))

    if use_projection:
        T = solve_symmetric_sylvester_kron(A, B, C)
        condM = cond(A)
    else:
        T = solve_sylvester(A, B, C)
        condM = cond(A)

    R = A @ T + T @ B - C
    resid = np.linalg.norm(R, 'fro')
    sym_err = np.linalg.norm(T - T.T, 'fro')

    if debug:
        solver_name = "Projection" if use_projection else "SciPy"
        print(f"[Oldroyd-B-{solver_name}] Sample {sample_idx} Residual={resid:.3e}, SymErr={sym_err:.3e}")

    if return_cond:
        return T, condM, resid
    else:
        return T