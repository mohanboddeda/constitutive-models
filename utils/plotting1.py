import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
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

    # Add German labels
    plt.text(1.1, 0.25, "einachsige Dehnung", fontsize=10, rotation=20,
             ha='center', va='center', backgroundcolor='w')

    plt.text(0.8, 0.02, "Schichtenströmungen\n+ ebene Strömungen", fontsize=10,
             ha='center', va='center', backgroundcolor='w')

    plt.text(1.1, -0.25, "äquibiaxiale Dehnung", fontsize=10, rotation=-20,
             ha='center', va='center', backgroundcolor='w')
    
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
###############################################################################################################

###############################################################################################################

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
###############################################################################################################

###############################################################################################################
def plot_carreau_parameter_study(gamma_list, nu_vals, nu_0, nu_inf, param_sets, save_path):
    """
    Plot Carreau–Yasuda viscosity curves for different parameter sets
    over a wide shear rate range, in non-dimensionalized form (nu/nu0),
    along with synthetic data points.
    """

    # === 1) Fixed shear rate range for full curve visualization ===
    shear_rates = np.logspace(-3, 9, 300)  # from 1e-3 to 1e9 1/s

    # === 2) Scale synthetic data to non-dimensional form ===
    nu_vals_nd = nu_vals / nu_0   # ν/ν₀

    # === 3) Create figure ===
    plt.figure(figsize=(8, 6))

    # Plot viscosity curves for each parameter set
    for n_val, lam_val, a_val in param_sets:
        nu_curve = nu_inf + (nu_0 - nu_inf) * (1 + (lam_val * shear_rates) ** a_val) ** ((n_val - 1) / a_val)
        nu_curve_nd = nu_curve / nu_0  # non-dimensional viscosity
        label = rf"$n={n_val}$, $\lambda={lam_val}$, $a={a_val}$"
        plt.loglog(shear_rates, nu_curve_nd, label=label, linewidth=2)

    # Scatter plot for synthetic data points
    plt.loglog(gamma_list, nu_vals_nd, 'o',
               alpha=0.5, markersize=5, label="Synthetic data", color='brown')

    # === 4) Axis settings ===
    plt.xlabel(r"Shear rate $\dot{\gamma}$ [s$^{-1}$]", fontsize=14)
    plt.ylabel(r"Reduced viscosity $\nu / \nu_0$ [-]", fontsize=14)

    # Log scale ticks for nice spacing
    plt.gca().xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    plt.gca().yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))

    plt.title("Carreau–Yasuda: Reduced viscosity vs Shear rate", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    # Save high-resolution image
    plt.savefig(save_path, dpi=300)
    plt.close()
##########################################################################################################

##########################################################################################################
def analyze_and_plot_dataset(X_flat, Y_flat, cfg):
    """
    Performs comprehensive statistical analysis and generates audience-ready plots.

    Saves results into per-model subfolders inside:
        images/analysis_output/<model_name>/<mode>/seed_<seed>/

    Notes:
    ------
    - cfg.mode: 'single_stage' or 'multi_stage'
    - cfg.seed: integer seed for this run
    """

    print("\n" + "="*50)
    print("📊 GENERATING DATA VISUALIZATIONS (via plotting.py)")
    print("="*50)

    # Mode string instead of removed 'stable/unstable'
    mode_str = cfg.mode  # 'single_stage' or 'multi_stage'
    run_id = f"{cfg.constitutive_eq}_{mode_str}_seed{cfg.seed}"

    # Create per-model folder inside analysis_output base
    model_folder_name = cfg.constitutive_eq  # e.g., "maxwell_B", "oldroyd_B", ...
    analysis_dir_base = os.path.join(cfg.paths.images, "analysis_output")

    # Full output directory includes mode and seed
    analysis_dir = os.path.join(analysis_dir_base, model_folder_name, mode_str, f"seed_{cfg.seed}")
    os.makedirs(analysis_dir, exist_ok=True)
    print(f"📂 Saving analysis output to: {analysis_dir}")

    # Each model's stats text file – separate per mode/seed
    stats_file_path = os.path.join(analysis_dir, "dataset_statistics_summary.txt")

    with open(stats_file_path, "a") as f:
        f.write(f"\n\n{'='*20} REPORT: {run_id} {'='*20}\n")

        def _process_variable(data_flat, var_name, var_type):
            mean_val = np.mean(data_flat)
            std_val  = np.std(data_flat)
            min_val  = np.min(data_flat)
            max_val  = np.max(data_flat)
            count_val = data_flat.size
            q1_val = np.percentile(data_flat, 25)
            median_val = np.percentile(data_flat, 50)
            q3_val = np.percentile(data_flat, 75)
            iqr_val = q3_val - q1_val

            # Write stats
            f.write(f"--- {var_name} ---\n")
            f.write(f"  Shape:      {data_flat.shape}\n")
            f.write(f"  Count:      {count_val}\n")
            f.write(f"  Mean:       {mean_val:.4e}\n")
            f.write(f"  Std Dev:    {std_val:.4e}\n")
            f.write(f"  Min:        {min_val:.4e}\n")
            f.write(f"  25% (Q1):   {q1_val:.4e}\n")
            f.write(f"  Median(Q2): {median_val:.4e}\n")
            f.write(f"  75% (Q3):   {q3_val:.4e}\n")
            f.write(f"  IQR:        {iqr_val:.4e}\n")
            f.write(f"  Max:        {max_val:.4e}\n")
            f.write(f"  Range:      [{min_val:.4e}, {max_val:.4e}]\n")

            filename_suffix = f"{var_type}_{run_id}"

            # Save histogram
            plt.figure(figsize=(10, 6))
            plt.hist(data_flat.flatten(), bins=50, color='steelblue', edgecolor='white', log=True)
            plt.title(f"Distribution of {var_name} ({run_id})")
            plt.xlabel(f"{var_name} Value")
            plt.ylabel("Frequency (Log Scale)")
            plt.grid(True, which="both", ls="--", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, f"hist_log_{filename_suffix}.png"))
            plt.close()

            # Save boxplot
            plt.figure(figsize=(8, 6))
            plt.boxplot(data_flat.flatten(), vert=True, patch_artist=True,
                        boxprops=dict(facecolor="lightblue"))
            plt.title(f"Box Plot: {var_name} ({run_id})")
            plt.ylabel("Value")
            plt.grid(True, ls="--", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, f"boxplot_{filename_suffix}.png"))
            plt.close()

        # Analysis for L and T
        _process_variable(X_flat, "Velocity Gradient (L)", "X")
        _process_variable(Y_flat, "Stress Tensor (T)", "Y")

    print(f"✅ Added stats to: {stats_file_path}")
    print(f"✅ Saved plots for {run_id}")

 