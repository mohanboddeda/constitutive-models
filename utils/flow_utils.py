import numpy as np
from utils.tensors import generate_random_rotation_matrix

def get_flow_eigenvalues(flow_type, rate=1.0):
    """
    Return canonical eigenvalues for specific rheological flow types.
    
    These eigenvalues define the shape of the deformation (Rate-of-Deformation tensor D)
    in its principal frame.
    
    Parameters:
    -----------
    flow_type : str
        Name of the flow (e.g., "uniaxial_extension", "pure_shear").
    rate : float
        The magnitude (strain rate) of the flow.
        
    Returns:
    --------
    list
        [lambda1, lambda2, lambda3] eigenvalues for D.
    """
    # 1. Uniaxial Extension: Stretching along one axis, compressing others to maintain volume (trace=0).
    # lambda = [rate, -rate/2, -rate/2]
    if flow_type == "uniaxial_extension":
        return [rate, -0.5 * rate, -0.5 * rate]
    
    # 2. Biaxial Extension: Stretching along two axes equally, compressing the third.
    # lambda = [rate, rate, -2*rate]
    elif flow_type == "biaxial_extension":
        return [rate, rate, -2.0 * rate]
    
    # 3. Planar Extension: Stretching x, Compressing y, z fixed (0).
    # lambda = [rate, -rate, 0]
    elif flow_type == "planar_extension":
        return [rate, -rate, 0.0]
    
    # 4. Pure Shear: 2D shear flow characteristic (similar to planar but rotated 45 deg in some contexts).
    # lambda = [rate/2, -rate/2, 0]
    elif flow_type == "pure_shear":
        return [0.5 * rate, -0.5 * rate, 0.0]
    
    # 5. Mixed Flows: Non-standard combinations often used to test robustness.
    elif flow_type == "mixed_flow_above":
        return [rate, -0.25 * rate, -0.75 * rate]
    
    elif flow_type == "mixed_flow_below":
        return [rate, 0.6 * rate, -1.6 * rate]
    
    else:
        raise ValueError(f"Unknown flow type: {flow_type}")


def generate_flow_L(flow_type, rate, dim=3, vorticity_ratio=1.0):
    """
    Generates a Velocity Gradient Tensor L = D + W for a specific flow type.
    
    The tensor is randomly rotated to ensure the dataset is not biased 
    to the canonical axes (x,y,z), making the network learn frame invariance.

    Parameters:
    -----------
    flow_type : str
        The target flow classification (e.g. 'uniaxial_extension').
    rate : float
        The magnitude of the deformation (derived from random sampling in main loop).
    dim : int
        Dimension of the tensor (usually 3).
    vorticity_ratio : float, optional (Default=1.0)
        Controls the magnitude of rotation relative to deformation.
        Passed from cfg.max_vorticity_ratio in the main script.
        
    Returns:
    --------
    np.ndarray
        The constructed L tensor (dim, dim).
    """
    # 1. Get Canonical Eigenvalues for D based on the physics of the flow type
    evals = get_flow_eigenvalues(flow_type, rate)
    
    # Ensure correct dimension (pad with zeros if needed, though usually 3D)
    if len(evals) < dim:
        evals = evals + [0.0] * (dim - len(evals))
    evals = evals[:dim]

    # 2. Construct D (Rate of Deformation)
    # We rotate it randomly so the flow direction is arbitrary in 3D space
    Lambda = np.diag(evals)
    R = generate_random_rotation_matrix(dim)
    D = R @ Lambda @ R.T
    
    # 3. Construct W (Vorticity)
    # Generate random vorticity axis vector
    w_vec = np.random.randn(dim)
    W = np.zeros((dim, dim))
    
    if dim == 3:
        # Fill upper triangle from w-vector then antisymmetrize
        W[0, 1], W[0, 2], W[1, 2] = -w_vec[2], w_vec[1], -w_vec[0]
        W = W - W.T
    elif dim == 2:
        W[0, 1] = -w_vec[0]
        W[1, 0] = w_vec[0]

    # 4. Scale W based on D and the requested vorticity_ratio
    # This ensures the rotational component is proportional to the deformation rate
    norm_D = np.linalg.norm(D)
    norm_W = np.linalg.norm(W)
    
    if norm_W > 1e-10:
        W_scaled = W * (vorticity_ratio * norm_D / norm_W)
    else:
        W_scaled = W
        
    # 5. Combine symmetric (D) and antisymmetric (W) parts to form L
    L = D + W_scaled
    
    return L