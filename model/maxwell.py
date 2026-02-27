import numpy as np
from numpy.linalg import cond
from scipy.linalg import solve_sylvester
from utils.solver_projection import solve_symmetric_sylvester_kron  # optional projection solver

def solve_steady_state_maxwell(L, eta0, lam,
                               return_cond=False, debug=False, sample_idx=None,
                               use_projection=False):
    """
    Solve steady-state Maxwell-B (upper-convected) equation:
        T - lam * (L*T + T*L^T) = 2*eta0*D
    
    This comes directly from the lecture notes steady-state form.
    Parameters
    ----------
    L : ndarray (3x3)
        Velocity gradient tensor
    eta0 : float
        Zero-shear viscosity
    lam : float
        Relaxation time
    use_projection : bool
        If True, use projection solver to enforce symmetry
    """
    # Ensure double precision
    L = L.astype(np.float64)
    dim = L.shape[0]
    
    # Symmetric part
    D = 0.5 * (L + L.T)
    
    # Sylvester equation form
    A = np.eye(dim) - lam * L
    B = -lam * L.T
    C = 2.0 * eta0 * D

    # Solve
    if use_projection:
        T = solve_symmetric_sylvester_kron(A, B, C)
        condM = cond(A)
    else:
        T = solve_sylvester(A, B, C)
        condM = cond(A)
    
    # Residuals
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