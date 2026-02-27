import os
import numpy as np
import torch
import flax
from sklearn.model_selection import train_test_split
import hydra
from omegaconf import DictConfig

# List of flow types
FLOW_TYPES = [
    "uniaxial_extension",
    "biaxial_extension",
    "planar_extension",
    "pure_shear",
    "mixed_flow_above",
    "mixed_flow_below"
]

def preprocess_and_save(flow_type, seed=42, scaling_mode="standard", dim=3, stable=True, model_type="maxwell_B"):
    """
    Preprocess dataset for a given flow type, loading .pt files instead of .npy
    and saving AFTER-NORMALIZATION stats to the dataset_statistics_summary.txt file
    inside the plots/analysis_output/<model_type>/ folder for that flow type.
    """
    folder_path = os.path.join("flow_data", flow_type)

    # choose suffix based on stability flag
    suffix = "_stable" if stable else "_unstable"

    # filenames follow your Maxwell-B saving convention
    X_file = f"X_{dim}D_{model_type}{suffix}.pt"
    Y_file = f"Y_{dim}D_{model_type}{suffix}.pt"

    X_path = os.path.join(folder_path, X_file)
    Y_path = os.path.join(folder_path, Y_file)

    if not os.path.exists(X_path) or not os.path.exists(Y_path):
        print(f"❌ Skipping {flow_type}: .pt data files not found.")
        return

    # Load tensors and convert to numpy
    X = torch.load(X_path).numpy()
    Y = torch.load(Y_path).numpy()
    print(f"\n📂 PreProcessing {flow_type} → X shape: {X.shape}, Y shape: {Y.shape}")

    # === Split train/val/test ===
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=seed, shuffle=True
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_val, Y_train_val, test_size=0.25, random_state=seed
    )

    print(f"📊 Split sizes: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")

    # === Normalize X ===
    X_mean = X_train.mean(axis=0)
    X_std  = X_train.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_train_n = (X_train - X_mean) / X_std
    X_val_n   = (X_val - X_mean) / X_std
    X_test_n  = (X_test - X_mean) / X_std

    # === Normalize Y ===
    if scaling_mode == "standard":
        Y_mean = Y_train.mean(axis=0)
        Y_std  = Y_train.std(axis=0)
        Y_std[Y_std == 0] = 1.0
        Y_train_n = (Y_train - Y_mean) / Y_std
        Y_val_n   = (Y_val   - Y_mean) / Y_std
        Y_test_n  = (Y_test  - Y_mean) / Y_std
    elif scaling_mode == "minmax":
        Y_min = Y_train.min(axis=0)
        Y_max = Y_train.max(axis=0)
        Y_range = np.where((Y_max - Y_min) == 0, 1.0, Y_max - Y_min)
        Y_train_n = (Y_train - Y_min) / Y_range
        Y_val_n   = (Y_val   - Y_min) / Y_range
        Y_test_n  = (Y_test  - Y_min) / Y_range
        Y_mean, Y_std = Y_min, Y_range
    else:
        raise ValueError("scaling_mode must be 'standard' or 'minmax'")

    # === AFTER NORMALIZATION STATS (saved only to file) ===
    def dataset_stats(name, arr):
        q1 = np.percentile(arr, 25)
        q2 = np.percentile(arr, 50)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        return (
            f"--- {name} ---\n"
            f"  Shape:      {arr.shape}\n"
            f"  Count:      {arr.size}\n"
            f"  Mean:       {np.mean(arr): .4e}\n"
            f"  Std Dev:    {np.std(arr): .4e}\n"
            f"  Min:        {np.min(arr): .4e}\n"
            f"  25% (Q1):   {q1: .4e}\n"
            f"  Median(Q2): {q2: .4e}\n"
            f"  75% (Q3):   {q3: .4e}\n"
            f"  IQR:        {iqr: .4e}\n"
            f"  Max:        {np.max(arr): .4e}\n"
            f"  Range:      [{np.min(arr): .4e}, {np.max(arr): .4e}]\n"
        )

    stats_text = []
    stats_text.append(f"\n================ AFTER NORMALIZATION ({model_type}) ================\n")
    stats_text.append(dataset_stats("Velocity Gradient (L) - Train Norm", X_train_n))
    stats_text.append(dataset_stats("Velocity Gradient (L) - Val Norm", X_val_n))
    stats_text.append(dataset_stats("Velocity Gradient (L) - Test Norm", X_test_n))
    stats_text.append(dataset_stats("Stress Tensor (T) - Train Norm", Y_train_n))
    stats_text.append(dataset_stats("Stress Tensor (T) - Val Norm", Y_val_n))
    stats_text.append(dataset_stats("Stress Tensor (T) - Test Norm", Y_test_n))

    # Path to plots/analysis_output/<model_type>
    plots_dir = os.path.join(folder_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    stats_file = os.path.join(plots_dir, "dataset_statistics_normalization.txt")

    # Append AFTER NORMALIZATION stats to the file
    with open(stats_file, "a") as f:
        f.write("".join(stats_text))

    print(f"📄 AFTER NORMALIZATION stats saved to: {stats_file}")

    # === Save splits and params ===
    np.save(os.path.join(folder_path, "X_train.npy"), X_train_n)
    np.save(os.path.join(folder_path, "X_val.npy"),   X_val_n)
    np.save(os.path.join(folder_path, "X_test.npy"),  X_test_n)
    np.save(os.path.join(folder_path, "Y_train.npy"), Y_train_n)
    np.save(os.path.join(folder_path, "Y_val.npy"),   Y_val_n)
    np.save(os.path.join(folder_path, "Y_test.npy"),  Y_test_n)
    np.save(os.path.join(folder_path, "X_mean.npy"), X_mean)
    np.save(os.path.join(folder_path, "X_std.npy"),  X_std)
    np.save(os.path.join(folder_path, "Y_mean.npy"), Y_mean)
    np.save(os.path.join(folder_path, "Y_std.npy"),  Y_std)

    return X_train_n, X_val_n, X_test_n, Y_train_n, Y_val_n, Y_test_n, X_mean, X_std, Y_mean, Y_std


def save_checkpoint(params, X_mean, X_std, Y_mean, Y_std, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    to_save = {"params": params, "X_mean": X_mean, "X_std": X_std, "Y_mean": Y_mean, "Y_std": Y_std}
    with open(path, "wb") as f:
        f.write(flax.serialization.to_bytes(to_save))


def load_checkpoint(path, init_params):
    with open(path, "rb") as f:
        restored = flax.serialization.from_bytes(init_params, f.read())
    return restored


@hydra.main(config_path="config/data", config_name="maxwell_config", version_base=None)
def main(cfg: DictConfig):
    """
    Run preprocessing using Hydra config.
    """
    flow_types = [cfg.flow_type] if isinstance(cfg.flow_type, str) else cfg.flow_type
    for ft in flow_types:
        preprocess_and_save(
            flow_type=ft,
            seed=cfg.seed,
            scaling_mode="standard",  # or cfg.scaling_mode if you add to config
            dim=3,
            stable=True,
            model_type=cfg.model_type if "model_type" in cfg else "maxwell_B"
        )


if __name__ == "__main__":
    main()