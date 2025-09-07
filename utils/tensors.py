import numpy as np

def generate_random_rotation_matrix(dim=3):
    H = np.random.randn(dim, dim)
    Q, _ = np.linalg.qr(H)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q

def generate_base_tensor(eigenvalues):
    dim = len(eigenvalues)
    eigenvalues = np.random.permutation(eigenvalues)
    R = generate_random_rotation_matrix(dim)
    Lambda = np.diag(eigenvalues)
    return R @ Lambda @ R.T

def generate_base_L_tensor(dim=3, vorticity_ratio=1.0):
    ew = np.random.randn(dim)
    ew -= np.mean(ew)
    D0 = generate_base_tensor(ew)
    w = np.random.randn(dim)
    W0 = np.zeros((dim, dim))
    if dim == 3:
        W0[0, 1], W0[0, 2], W0[1, 2] = -w[2], w[1], -w[0]
        W0 = W0 - W0.T
    elif dim == 2:
        W0[0, 1] = -w[0]
        W0[1, 0] = w[0]
    norm_D0 = np.linalg.norm(D0)
    norm_W0 = np.linalg.norm(W0)
    if norm_W0 > 1e-10:
        W0_scaled = W0 * (vorticity_ratio * norm_D0 / norm_W0)
    else:
        W0_scaled = W0
    L0 = D0 + W0_scaled
    return L0

def flatten_symmetric_tensors(tensors):
    dim = tensors.shape[-1]
    if dim == 3:
        return np.stack([
            tensors[..., 0, 0], tensors[..., 1, 1], tensors[..., 2, 2],
            tensors[..., 0, 1], tensors[..., 0, 2], tensors[..., 1, 2]
        ], axis=-1)
    elif dim == 2:
        return np.stack([
            tensors[..., 0, 0], tensors[..., 1, 1], tensors[..., 0, 1]
        ], axis=-1)
    else:
        raise ValueError(f"Flattening für Dimension {dim} ist nicht implementiert.")