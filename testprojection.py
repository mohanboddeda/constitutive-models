import numpy as np
from scipy.linalg import solve_sylvester

def build_selection_matrix(n):
    """Build symmetric selection matrix with orthonormal columns."""
    m = n*(n+1)//2
    S = np.zeros((n*n, m))
    col = 0
    for i in range(n):
        for j in range(i, n):
            e = np.zeros((n, n))
            if i == j:
                e[i, j] = 1.0
            else:
                e[i, j] = 1.0
                e[j, i] = 1.0
                e /= np.sqrt(2)       # scale off-diagonal by 1/sqrt(2)
            S[:, col] = e.flatten()
            col += 1
    return S

def solve_symmetric_sylvester_kron(A, B, C):
    n = A.shape[0]
    K = np.kron(np.eye(n), A) + np.kron(B.T, np.eye(n))
    c_full = C.flatten()

    S = build_selection_matrix(n)
    M_sym = S.T @ K @ S
    c_sym = S.T @ c_full

    t_sym = np.linalg.solve(M_sym, c_sym)
    t_full = S @ t_sym
    T = t_full.reshape(n, n)
    return T

# ==== Test vs SciPy ====
np.random.seed(1)
L = np.random.rand(3, 3)
D = 0.5*(L + L.T)
lam = 1.902
eta = 5.28e-5

A = (1/lam) * np.eye(3) - L.T
B = -L
C = (2.0 * eta / lam) * D

T_scipy = solve_sylvester(A, B, C)
T_proj = solve_symmetric_sylvester_kron(A, B, C)

# Residuals
resid_scipy = np.linalg.norm(A@T_scipy + T_scipy@B - C, 'fro')
resid_proj = np.linalg.norm(A@T_proj + T_proj@B - C, 'fro')
diff_norm = np.linalg.norm(T_scipy - T_proj, 'fro')

print("SciPy Residual:", resid_scipy)
print("Proj Residual :", resid_proj)
print("Diff Norm     :", diff_norm)
print("Projection SymErr:", np.linalg.norm(T_proj - T_proj.T))