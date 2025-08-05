import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import hydra
from   omegaconf import DictConfig
from   scipy.linalg import solve_sylvester
from pathlib import Path

# =============================================================================
# 1. TENSOR-ERZEUGUNG
# =============================================================================

def generate_random_rotation_matrix(dim=3):
    """Erzeugt eine zufällige Rotationsmatrix."""
    H = np.random.randn(dim, dim)
    Q, _ = np.linalg.qr(H)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q

def generate_base_tensor(eigenvalues):
    """Erzeugt einen symmetrischen Basistensor D0 aus gegebenen Eigenwerten."""
    dim = len(eigenvalues)
    eigenvalues = np.random.permutation(eigenvalues)
    R = generate_random_rotation_matrix(dim)
    Lambda = np.diag(eigenvalues)
    return R @ Lambda @ R.T

def generate_base_L_tensor(dim=3, vorticity_ratio=1.0):
    """
    Generiert einen physikalisch sinnvollen, spurfreien Basis-Geschwindigkeitsgradienten L₀.
    Kombiniert einen Deformationsteil D₀ und einen Rotationsteil W₀.
    """
    # Eigenwerte so wählen, dass die Spur von D₀ null ist (Inkompressibilität)
    ew = np.random.randn(dim)
    ew -= np.mean(ew)

    # 1. Erzeuge den symmetrischen Anteil D₀
    D0 = generate_base_tensor(ew)

    # 2. Erzeuge einen zufälligen anti-symmetrischen Anteil W₀
    w = np.random.randn(dim)
    W0 = np.zeros((dim, dim))
    if dim == 3:
        W0[0, 1], W0[0, 2], W0[1, 2] = -w[2], w[1], -w[0]
        W0 = W0 - W0.T  # Füllt die untere Dreiecksmatrix symmetrisch
    elif dim == 2:
        W0[0, 1] = -w[0]
        W0[1, 0] = w[0]

    # 3. Skaliere W₀ basierend auf der Vorticity Ratio
    norm_D0 = np.linalg.norm(D0)
    norm_W0 = np.linalg.norm(W0)
    
    if norm_W0 > 1e-10:
        W0_scaled = W0 * (vorticity_ratio * norm_D0 / norm_W0)
    else:
        W0_scaled = W0 # Falls W0 zufällig null ist

    # 4. Setze L₀ zusammen
    L0 = D0 + W0_scaled
    return L0

# =============================================================================
# 2. DATENHANDHABUNG & PLOTTING
# =============================================================================

def flatten_tensors_vectorized(tensors):
    """Wandelt einen Stapel von Tensoren in flache Vektoren um (n,n) -> (n*n)."""
    return tensors.reshape(tensors.shape[0], tensors.shape[1], -1)

def flatten_symmetric_tensors(tensors):
    """Reduziert einen Stapel symmetrischer Tensoren auf ihre einzigartigen Komponenten."""
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

    
def compute_invariants_vectorized(D):
    if D.ndim == 2: D = D[np.newaxis, :, :]
    I = np.trace(D, axis1=-2, axis2=-1)
    II = 0.5 * (I**2 - np.trace(D @ D, axis1=-2, axis2=-1))
    III = np.linalg.det(D) if D.shape[-1] == 3 else np.zeros(D.shape[0])
    return I, II, III

def plot_invariants_diagram(cfg, L0_list, title="Strömungsart im Invariantenraum", paths=False):
    D0_list = [0.5 * (L0 + L0.T) for L0 in L0_list]
    D0_array = np.array(D0_list) 

    dim = D0_array.shape[-1]
    plt.figure(figsize=(8, 6))
    
    # Hintergrund (Diskriminantenbereich)
    II_vals = np.linspace(0, 1.5, 300)
    III_vals = np.linspace(-0.5, 0.5, 300)
    II_grid, III_grid = np.meshgrid(II_vals, III_vals)
    discriminant = ((-III_grid / 2)**2 + (-II_grid / 3)**3)
    plt.contourf(II_grid, III_grid, discriminant, levels=[-1e10, 0, 1e10], colors=['#ffa8a8', '#a8c6ff'], alpha=0.8)
    
    # 2. Zeichne die Startpunkte (D0)
    I_D0, II_D0, III_D0 = compute_invariants_vectorized(D0_array)

    # Plot invariant domain boundary curves with thicker lines
    II_boundary = np.linspace(0, 1.5, 500)
    III_boundary = 2 * np.sqrt(np.maximum(0, - (II_boundary / 3) ** 3))  # safe sqrt

    plt.plot(II_boundary, III_boundary, 'k-', linewidth=2, label='Boundary')
    plt.plot(II_boundary, -III_boundary, 'k-', linewidth=2)
    
    # Alle D0 haben Spur 0 und werden rot gezeichnet
    plt.scatter(-II_D0, III_D0, color="red", s=40, label="Steady-State D0",edgecolors='k', linewidth=0.7)

    plt.xlabel("-II (Zweite Invariante)")
    plt.ylabel("III (Dritte Invariante)")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    # Achsen-Limits an typische Werte für normierte Eigenwerte anpassen
    plt.xlim(0, 1.5)
    plt.ylim(-0.5, 0.5)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.tight_layout()
    filename = os.path.join(cfg.paths.images, f"{cfg.dim}D_Invarianten_Diagramm_{paths}.png")
    plt.savefig(filename, dpi=300)
    plt.close()

# =============================================================================
# 3.  STEADY STATE MAXWELL-MODELL SOLVER
# =============================================================================
# Maxwell B Model: Steady-State Constitutive Equation
# The governing equation in tensor form:
# T - λ * L^T * T - λ * T * L = 2 * η_0 * D
# where,
# T: Current stress tensor (unknown to solve)
# λ (lam): Relaxation time of the material
# L: Velocity gradient tensor (known input)
# L^T: Transpose of L
# D: Rate of deformation tensor, defined as D = 0.5 * (L + L^T)
# η_0 (eta0): Zero-shear viscosity

# Rearranged as a Sylvester equation for solving T:
#
# (I - λ * L^T) * T - T * (λ * L) = 2 * η_0 * D
#
# Define:
# A = I - λ * L^T
# B = λ * L
# C = 2 * η_0 * D
#
# Then solve the matrix equation:
# A * T - T * B = C

def solve_steady_state_maxwell(L, eta0=5.28e-5, lam=1.902):
    """
     Solve steady-state upper-convected Maxwell model (Maxwell-B):
    (1/λ*I - L^T) T - T (L) = 2*η0/λ*D
    
    Args:
        L : (dim x dim) velocity gradient tensor
        eta0 : zero shear viscosity
        lam : relaxation time
    
    Returns:
        T : stress tensor
    """
    # Convert L to double precision for better numerical accuracy
    L = L.astype(np.float64)
    dim = L.shape[0]
    D = 0.5 * (L + L.T)
    A = (1/lam) * np.eye(dim) - L.T
    B = L
    C = (2 * eta0 / lam) * D
    # Solve the matrix equation: A T - T B = C
    T = solve_sylvester(A, B, C)
    # Enforce symmetry just before returning
    T = 0.5 * (T + T.T)
    return T


# =============================================================================
# 4.  STEADY STATE OLDROYD-B SOLVER
# =============================================================================   
def solve_steady_state_oldroyd(L, eta0=5.28e-5, lam=1.902, lam_r=1.0):
    """
    Solve steady-state kontravariant Oldroyd-B model:
    (I - λ L) T - T (λ L^T) = 2 η0 [D - λ_r L D - λ_r D L^T]
    
    Args:
        L : (dim x dim) velocity gradient tensor
        eta0 : zero shear viscosity
        lam : relaxation time
        lam_r : retardation time
    
    Returns:
        T : stress tensor
    """
    L = L.astype(np.float64)
    dim = L.shape[0]
    D = 0.5 * (L + L.T)

    A = np.eye(dim) - lam * L
    B = lam * L.T
    C = 2 * eta0 * (D - lam_r * (L @ D) - lam_r * (D @ L.T))

    T = solve_sylvester(A, B, C)
    # Ensure symmetric result (remove numerical asymmetry)
    T = 0.5 * (T + T.T)
    return T

# =============================================================================
# 5.  PLOTTING SHEAR STRESS TENSORS T 
# =============================================================================  


def plot_tensor_matrix(tensor, model_name="Model", title=None, filename=None):
    """
    Plot a 2D tensor as a heatmap with annotations.
    
    Args:
        tensor (np.ndarray): 2D square tensor to plot
        model_name (str): Name of the model, e.g., 'Maxwell-B' or 'Oldroyd-B'
        title (str, optional): Custom plot title. If None, it defaults to '{model_name} Stress Tensor'
        filename (str, optional): Path to save the plot image. If None, plot is shown interactively.
    """
    
    
    if title is None:
        title = f"{model_name} Stress Tensor"
    
    plt.figure(figsize=(6, 6))
    plt.imshow(tensor, cmap='viridis', interpolation='nearest')
    
    # Annotate matrix entries
    for (i, j), val in np.ndenumerate(tensor):
        plt.text(j, i, f"{val:.2e}", ha='center', va='center', color='w', fontsize=10)
    
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    
    if filename:
        # Insert model name in filename if not already present
        if model_name.lower() not in filename.lower():
            base, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = f"{base}_{model_name}.{ext}" if ext else f"{base}_{model_name}"
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()
        plt.close()

# =============================================================================
# 6. CARREAU YASUDA MODELL SOLVER
# =============================================================================

def carreau_yasuda_viscosity(L, nu_0, nu_inf, lambda_val, n, a):
    """
    Berechnet die stationäre (steady-state) Viskosität nach dem Carreau-Yasuda-Modell
    für ein gegebenes Geschwindigkeitsgradient-Tensor L.
    
    Args:
        L (np.ndarray): (dim, dim) velocity gradient tensor.
        nu_0 (float): Zero-shear (low shear) viscosity.
        nu_inf (float): Infinite-shear (high shear) viscosity.
        lambda_val (float): Relaxation time.
        n (float): Power-law index.
        a (float): Yasuda parameter (transition sharpness).
        
    Returns:
        float: Steady-state viscosity (scalar)
    """
    D = 0.5 * (L + L.T)
    _, second_invariant_D, _ = compute_invariants_vectorized(D)
    second_invariant_D = -second_invariant_D  # Make it positive-definite
    epsilon = 1e-12 # epsilon prevents taking square roots of zero
    shear_rate = 2 * np.sqrt(second_invariant_D + epsilon)[0]  # [0] extracts the scalar from a single-element array or similar structure
    term1 = (lambda_val * shear_rate) ** a      #here might be some issue 
    term2 = (1 + term1) ** ((n - 1) / a)        #here might be some issue 
    nu = nu_inf + (nu_0 - nu_inf) * term2       #here might be some issue 
    return nu


def carreau_viscosity_shearrate(gamma_dot, nu_0, nu_inf, lam, n, a):
    return nu_inf + (nu_0 - nu_inf) * (1 + (lam * gamma_dot)**a)**((n - 1) / a)

def plot_carreau_parameter_study(gamma_list, nu_vals, nu_0, nu_inf, param_sets, save_path):
    """
    Plot Carreau-Yasuda viscosity curves for different parameter sets over shear rate,
    together with synthetic data points.

    Args:
        gamma_list (array-like): Shear rates from synthetic data
        nu_vals (array-like): Corresponding viscosities from synthetic data
        nu_0 (float): Zero-shear viscosity
        nu_inf (float): Infinite-shear viscosity
        param_sets (list of tuples): List of (n, lam, a) parameter tuples
        save_path (str): File path to save the plot image
    """
    
     # Generate shear rate vector spanning desired range
    shear_rates = np.logspace(-3, 9, 200)  # 10^-3 to 10^9, 200 points
    
    plt.figure(figsize=(8, 6))
    
    # Plot viscosity curves for each parameter set
    for n_val, lam_val, a_val in param_sets:
        nu_curve = carreau_viscosity_shearrate(shear_rates, nu_0, nu_inf, lam_val, n_val, a_val)
        label = f"n={n_val}, λ={lam_val}, a={a_val}"
        plt.loglog(shear_rates, nu_curve, label=label, marker='o', markevery=15)
    
    # Scatter plot of synthetic data
    plt.loglog(gamma_list, nu_vals, 'o', alpha=0.4, label="Synthetic data")

    # Set axis limits to cover wide range as in lecture
    plt.xlim(1e-3, 1e9)
    plt.ylim(1e-3, 1e4)
    
    plt.xlabel(r"Shear rate $\dot{\gamma}$ [s$^{-1}$]")
    plt.ylabel(r"Viscosity $\nu$ [Pa·s]")
    plt.title("Carreau–Yasuda: Viscosity vs Shear rate parameter study")
    plt.legend()
    plt.grid(True, which='both', linestyle=':')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# =============================================================================
# 7. HAUPTSKRIPT
# =============================================================================

@hydra.main(config_path="config/data", config_name="dataConfig", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Hauptfunktion: Generiert L und berechnet die Spannung τ bzw. ν,
    """
    # 1) Creates folders for images and data
    os.makedirs(cfg.paths.images, exist_ok=True)
    os.makedirs(cfg.paths.data,  exist_ok=True)
    L0_list, Y_list = [], []
    
    # Initialize nu_min and nu_max to track color scale if needed
    if cfg.constitutive_eq == "carreau_yasuda":
        nu_min, nu_max = float('inf'), float('-inf')

# 2) Single sampling loop for all 3 models
    for i in range(cfg.n_samples):
        # 2.1) sample L0
        v_ratio = np.random.uniform(0, cfg.max_vorticity_ratio)
        L0 = generate_base_L_tensor(dim=cfg.dim, vorticity_ratio=v_ratio)
        L0_list.append(L0)

        # 2.1) compute output based on constitutive_eq
        if cfg.constitutive_eq == "carreau_yasuda":
            nu = carreau_yasuda_viscosity(
                L0,
                nu_0=5.28e-5, nu_inf=3.30e-6,
                lambda_val=1.902, n=0.22, a=1.25,
            )
            Y_list.append(nu)
            # Update min/max for consistent color scale if plot needed
            nu_min = min(nu_min, nu)
            nu_max = max(nu_max, nu)

        elif cfg.constitutive_eq == "maxwell_B":
            T = solve_steady_state_maxwell(L0)
            Y_list.append(T)

        elif cfg.constitutive_eq == "oldroyd_B":
             T = solve_steady_state_oldroyd(L0, eta0=5.28e-5, lam=1.902, lam_r=1.0)
             Y_list.append(T)
        else:
            raise ValueError(f"Unknown constitutive_eq: {cfg.constitutive_eq}")

        # 2.2) Plotting inputs and outputs for all 3 modells

    for i in range(min(3, len(L0_list))):
        L0 = L0_list[i]
        plot_tensor_matrix(
            L0,
            title=f"L0 sample {i}",
            filename=os.path.join(cfg.paths.images, f"L0_sample_{i}.png"),
        )
        if cfg.constitutive_eq == "carreau_yasuda":
            nu = carreau_yasuda_viscosity(
                L0,
                nu_0=5.28e-5, nu_inf=3.30e-6,
                lambda_val=1.902, n=0.22, a=1.25,
            )
            print(f"Viscosity for sample {i}: {nu:.3e} Pa.s")
            plt.figure(figsize=(4, 4))
            plt.imshow(
                [[nu]],
                interpolation='nearest',
                vmin=nu_min, vmax=nu_max,
            )
            
            plt.colorbar(label="Viscosity ν")
            plt.title(f"Carreau-Yasuda ν sample {i}")
            plt.tight_layout()
            plt.savefig(os.path.join(cfg.paths.images, f"nu_sample_{i}.png"))
            plt.close()
        elif cfg.constitutive_eq == "maxwell_B":
            T = solve_steady_state_maxwell(L0)
            plot_tensor_matrix(
                T,
                model_name="Maxwell-B",
                title=f"Maxwell-B Stress Tensor sample {i}",
                filename=os.path.join(cfg.paths.images, f"T_maxwell_sample_{i}.png"),
            )
        elif cfg.constitutive_eq == "oldroyd_B":
            T = solve_steady_state_oldroyd(L0, eta0=5.28e-5, lam=1.902, lam_r=1.0)
            plot_tensor_matrix(
                T,
                model_name="Oldroyd-B",
                title=f"Oldroyd-B Stress Tensor sample {i}",
                filename=os.path.join(cfg.paths.images, f"T_oldroyd_sample_{i}.png"),
            )

    # 3) After loop: assemble arrays for saving
    X_np = np.array(L0_list)
    Y_np = np.array(Y_list)

    if cfg.constitutive_eq == "carreau_yasuda":
        X_flat = X_np.reshape(X_np.shape[0], -1)
        Y_flat = Y_np.reshape(-1, 1)
    elif cfg.constitutive_eq == "maxwell_B":
        X_flat = X_np.reshape(X_np.shape[0], -1)
        Y_flat = flatten_symmetric_tensors(Y_np)
    elif cfg.constitutive_eq == "oldroyd_B":
        X_flat = X_np.reshape(X_np.shape[0], -1)
        Y_flat = flatten_symmetric_tensors(np.array(Y_list))

    # 4) Invariants diagram (optional)
    plot_invariants_diagram(
        cfg,
        L0_list,
        title=f"{cfg.dim}D Strömungsarten im Invariantenraum",
        paths=True,
    )

    # 5) Save tensors
    data_path_X = os.path.join(cfg.paths.data, f"X_{cfg.dim}D_{cfg.constitutive_eq}.pt")
    data_path_Y = os.path.join(cfg.paths.data, f"Y_{cfg.dim}D_{cfg.constitutive_eq}.pt")
    torch.save(torch.tensor(X_flat, dtype=torch.float32), data_path_X)
    torch.save(torch.tensor(Y_flat, dtype=torch.float32), data_path_Y)

    # 6) Plot Carreau-Yasuda parameter study with synthetic data
    if cfg.constitutive_eq == "carreau_yasuda":
        gamma_list = []
        epsilon = 1e-12
        for L0 in L0_list:
            D = 0.5 * (L0 + L0.T)
            _, second_invariant_D, _ = compute_invariants_vectorized(D)
            second_invariant_D = -second_invariant_D  # Make it positive definite
            shear_rate = 2 * np.sqrt(second_invariant_D + epsilon)[0]  # take scalar
            gamma_list.append(shear_rate)

        nu_vals = np.array(Y_list).flatten()

        param_sets = [
            (0.25, 1.0, 2.0),
            (0.5, 1.0, 2.0),
            (0.25, 100.0, 2.0),
            (0.5, 100.0, 2.0),
            (0.25, 100.0, 0.5),
        ]

        save_path = os.path.join(cfg.paths.images, "carreau_parameter_study.png")

        plot_carreau_parameter_study(
            gamma_list,
            nu_vals,
            nu_0=5.28e-5,
            nu_inf=3.30e-6,
            param_sets=param_sets,
            save_path=save_path,
        )

    print("\nFertig")
    print(f"-> X shape: {X_flat.shape}")
    print(f"-> Y shape: {Y_flat.shape}")



if __name__ == "__main__":
    main()