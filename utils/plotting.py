import os
import numpy as np
import matplotlib.pyplot as plt
from utils.invariants import compute_invariants_vectorized

def plot_invariants_diagram(cfg, L0_list, title="Strömungsart im Invariantenraum", paths=False):
    D0_list = [0.5 * (L0 + L0.T) for L0 in L0_list]
    D0_array = np.array(D0_list) 

    plt.figure(figsize=(8, 6))
    
    # Background discriminant area
    II_vals = np.linspace(0, 1.5, 300)
    III_vals = np.linspace(-0.5, 0.5, 300)
    II_grid, III_grid = np.meshgrid(II_vals, III_vals)
    discriminant = ((-III_grid / 2)**2 + (-II_grid / 3)**3)
    plt.contourf(II_grid, III_grid, discriminant, levels=[-1e10, 0, 1e10], 
                 colors=['#ffa8a8', '#a8c6ff'], alpha=0.8)
    
    # Boundary curves
    II_boundary = np.linspace(0, 1.5, 500)
    III_boundary = 2 * np.sqrt(np.maximum(0, - (II_boundary / 3) ** 3))
    plt.plot(II_boundary, III_boundary, 'k-', linewidth=2, label='Boundary')
    plt.plot(II_boundary, -III_boundary, 'k-', linewidth=2)
    
    # Scatter steady-state points
    _, II_D0, III_D0 = compute_invariants_vectorized(D0_array)
    plt.scatter(-II_D0, III_D0, color="red", s=40, label="Steady-State D0", 
                edgecolors='k', linewidth=0.7)

    plt.xlabel("-II (Zweite Invariante)")
    plt.ylabel("III (Dritte Invariante)")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    plt.xlim(0, 1.5)
    plt.ylim(-0.5, 0.5)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.tight_layout()
    filename = os.path.join(cfg.paths.images, f"{cfg.dim}D_Invarianten_Diagramm_{paths}.png")
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_tensor_matrix(tensor, model_name="Model", title=None, filename=None):
    if title is None:
        title = f"{model_name} Stress Tensor"
    
    plt.figure(figsize=(6, 6))
    plt.imshow(tensor, cmap='viridis', interpolation='nearest')
    
    for (i, j), val in np.ndenumerate(tensor):
        plt.text(j, i, f"{val:.2e}", ha='center', va='center', color='w', fontsize=10)
    
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    
    if filename:
        if model_name.lower() not in filename.lower():
            base, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = f"{base}_{model_name}.{ext}" if ext else f"{base}_{model_name}"
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_carreau_parameter_study(gamma_list, nu_vals, nu_0, nu_inf, param_sets, save_path):
    """
    Plot Carreau-Yasuda viscosity curves for different parameter sets
    over shear rate, along with given data points.
    """
    # Generate shear rate vector
    shear_rates = np.logspace(np.log10(min(gamma_list)+ 1e-8), np.log10(max(gamma_list)), 200)
    
    plt.figure(figsize=(8, 6))
    
    # Plot viscosity curves for each parameter set
    for n_val, lam_val, a_val in param_sets:
        nu_curve = nu_inf + (nu_0 - nu_inf) * (1 + (lam_val * shear_rates) ** a_val) ** ((n_val - 1) / a_val)
        label = f"n={n_val}, λ={lam_val}, a={a_val}"
        plt.loglog(shear_rates, nu_curve, label=label)
    
    # Scatter plot for provided data
    plt.loglog(gamma_list, nu_vals, 'o', alpha=0.4, label="Synthetic data")

    plt.xlabel(r"Shear rate $\dot{\gamma}$ [s$^{-1}$]")
    plt.ylabel(r"Viscosity $\nu$ [Pa·s]")
    plt.title("Carreau–Yasuda: Viscosity vs Shear rate parameter study")
    plt.legend()
    plt.grid(True, which='both', linestyle=':')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()