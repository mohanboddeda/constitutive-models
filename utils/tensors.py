import numpy as np

#Generates a random orthonormal rotation matrix R (R^T.R=I, det R = 1)
#This is used to rotate tensors into random orientations in space — simulating different flow directions.

def generate_random_rotation_matrix(dim=3):
    H = np.random.randn(dim, dim)    # Random Gaussian matrix
    Q, _ = np.linalg.qr(H)           # QR decomposition → orthogonal Q
    if np.linalg.det(Q) < 0:         # Ensure proper righ hand rotation (determinant +1)
        Q[:, 0] *= -1
    return Q

#Given a set of eigenvalues (principal values of a tensor), constructs a symmetric tensor:
#T=RΛRT where:
#    Λ = diag(eigenvalues) , R = random rotation
# This ensures T is symmetric by construction and has those eigenvalues in a random frame.

def generate_base_tensor(eigenvalues):
    dim = len(eigenvalues)
    eigenvalues = np.random.permutation(eigenvalues)  # Random shuffle principal values 
    R = generate_random_rotation_matrix(dim)          # Random orientation
    Lambda = np.diag(eigenvalues)                     # Diagonal eigenvalue matrix
    return R @ Lambda @ R.T                           # Rotate Λ into full tensor

#Step 1 — Generate D0​ (symmetric strain rate tensor)
#Step 2 — Generate W0​ (antisymmetric vorticity tensor)
#Step 3 — Scale W0​ relative to D0​
#Step 4 — Combine into full L0=D0+W0​

def generate_base_L_tensor(dim=3, vorticity_ratio=1.0):
    ew = np.random.randn(dim)  # np.random.uniform(0.1, 2, dim) uniform principal values  #ew = np.random.randn(dim) #random principal values
    ew -= np.mean(ew)
    D0 = generate_base_tensor(ew)

    w = np.random.randn(dim)  # random components of vorticity vector
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

#For storage/lightweight arrays, takes symmetric
#3×3 tensor(s) and flattens to vec6:
#[Txx​,Tyy​,Tzz​,Txy​,Txz​,Tyz​] Perfect for saving .pt datasets later.

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