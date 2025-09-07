import numpy as np

def build_selection_matrix(n):
    """
    Builds the selection/expansion matrix S for symmetric matrices.
    Columns correspond to unique DOFs: diagonals then off-diagonals.
    Off-diagonals are scaled by 1/sqrt(2) for orthonormality.
    """
    m = n * (n + 1) // 2
    S = np.zeros((n * n, m))
    col = 0
    for i in range(n):
        for j in range(i, n):
            e = np.zeros((n, n))
            if i == j:
                e[i, j] = 1.0
            else:
                e[i, j] = 1.0
                e[j, i] = 1.0
                e /= np.sqrt(2)  # scaling for off-diagonals
            S[:, col] = e.flatten()
            col += 1
    return S

def solve_symmetric_sylvester_kron(A, B, C):
    """
    Solve A*T + T*B = C for symmetric T using
    Kronecker formulation restricted to symmetric subspace.
    """
    n = A.shape[0]
    # Full Kronecker matrix for Sylvester operator
    K = np.kron(np.eye(n), A) + np.kron(B.T, np.eye(n))
    c_full = C.flatten()

    # Build selection matrix
    S = build_selection_matrix(n)
    # Reduced system: Sᵀ K S t_sym = Sᵀ c_full
    M_sym = S.T @ K @ S
    c_sym = S.T @ c_full

    t_sym = np.linalg.solve(M_sym, c_sym)
    t_full = S @ t_sym
    T = t_full.reshape(n, n)

    return T