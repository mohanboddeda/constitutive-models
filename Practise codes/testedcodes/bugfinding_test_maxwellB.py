import numpy as np
from model.maxwell import solve_steady_state_maxwell
from utils.tensors import generate_base_L_tensor

# ---------------------------
# Settings for test
# ---------------------------
seed = 42
rng = np.random.default_rng(seed)
dim = 3

# eta0 and lam values to test scaling
eta0_values = [1, 10, 50]
lam_values = [1, 10]

# ---------------------------
# Step A: Generate sample L tensors
# ---------------------------
n_samples = 5
L_list = []
for i in range(n_samples):
    v_ratio = rng.uniform(0, 10)  # you can adjust sampling range
    L = generate_base_L_tensor(dim=dim, vorticity_ratio=v_ratio)
    L_list.append(L)

# ---------------------------
# Step B: Check D magnitudes
# ---------------------------
print("\n=== L and D magnitude check ===")
for i, L in enumerate(L_list):
    D = 0.5 * (L + L.T)
    print(f"\nSample {i}:")
    print(f"L Frobenius norm       = {np.linalg.norm(L, 'fro'):.4f}")
    print(f"D Frobenius norm       = {np.linalg.norm(D, 'fro'):.4f}")
    print("L:\n", np.round(L, 4))
    print("D:\n", np.round(D, 4))

# ---------------------------
# Step C: Test T scaling with eta0 and lam
# ---------------------------
print("\n=== T magnitude scaling test ===")
for i, L in enumerate(L_list):
    D = 0.5 * (L + L.T)
    print(f"\n--- Sample {i} ---")
    base_norms = {}
    for lam in lam_values:
        print(f"\nλ = {lam}")
        for eta0 in eta0_values:
            T, condM, resid = solve_steady_state_maxwell(L, eta0=eta0, lam=lam, return_cond=True)
            norm_T = np.linalg.norm(T, 'fro')
            norm_D = np.linalg.norm(D, 'fro')
            base_norms[(eta0, lam)] = norm_T
            print(f"η₀={eta0:<5} -> ‖T‖={norm_T:.4f}, cond={condM:.2e}, resid={resid:.2e}, ‖D‖={norm_D:.4f}")
        # Check scaling ratios
        ratio_10_1  = base_norms[(10, lam)] / base_norms[(1, lam)]
        ratio_50_1  = base_norms[(50, lam)] / base_norms[(1, lam)]
        print(f"Scaling η₀=1→10 ratio: {ratio_10_1:.4f} (should be ~10)")
        print(f"Scaling η₀=1→50 ratio: {ratio_50_1:.4f} (should be ~50)")

# ---------------------------
# Step D: Conclusion hints
# ---------------------------
print("\n=== How to read these results ===")
print("""
- If ‖D‖ is near zero for most samples: L generator problem (pure rotation or tiny gradient magnitudes).
- If ‖T‖ does not scale ~linearly with η₀ when L is fixed: solver or equation bug.
- If both L and T have reasonable magnitudes and scaling: data generation is physically consistent.
- Check condM (condition number): very high values (>1e8) mean numerical instability in solver.
""")