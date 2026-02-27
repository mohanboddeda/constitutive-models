import os
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from omegaconf import DictConfig
import hydra
from torch.utils.tensorboard import SummaryWriter
from tabulate import tabulate
from sklearn.metrics import mean_absolute_error
from utils.train_utils_maxwellB import plot_stress_tensor_comparison

from utils.train_utils_maxwellB import (
    load_and_normalize_data_maxwellB as load_and_normalize_data,
    save_checkpoint,
    load_checkpoint,
    plot_learning_curves,
    plot_residual_hist,
    plot_residuals_vs_pred
)

# --- Maxwell-B constants ---
ETA0 = 5.28e-5
LAM = 1.902

# --- Convert 6-vector to symmetric tensor ---
def vec6_to_sym3(vec):
    T = jnp.zeros((vec.shape[0], 3, 3))
    T = T.at[:, 0, 0].set(vec[:, 0])
    T = T.at[:, 1, 1].set(vec[:, 1])
    T = T.at[:, 2, 2].set(vec[:, 2])
    T = T.at[:, 0, 1].set(vec[:, 3]); T = T.at[:, 1, 0].set(vec[:, 3])
    T = T.at[:, 0, 2].set(vec[:, 4]); T = T.at[:, 2, 0].set(vec[:, 4])
    T = T.at[:, 1, 2].set(vec[:, 5]); T = T.at[:, 2, 1].set(vec[:, 5])
    return T

# --- Residual for Maxwell-B ---
def maxwellB_residual(L_phys, T_phys):
    D = 0.5 * (L_phys + jnp.swapaxes(L_phys, 1, 2))
    LTt = jnp.matmul(jnp.swapaxes(L_phys, 1, 2), T_phys)
    TL  = jnp.matmul(T_phys, L_phys)
    return T_phys - LAM * (LTt + TL) - 2.0 * ETA0 * D

# --- Activation functions map ---
activation_map = {
    "relu": nn.relu,
    "tanh": nn.tanh,
    "sigmoid": nn.sigmoid,
}

# --- MLP model ---
class MLP(nn.Module):
    features: list
    dropout: float = 0.0
    activation_fn: callable = nn.relu

    @nn.compact
    def __call__(self, x, train=True):
        if x.ndim == 3:
            x = x.reshape((x.shape[0], -1))
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = self.activation_fn(x)
            x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
        return nn.Dense(self.features[-1])(x)

# --- Loss function ---
def compute_losses(params, model, x_norm, y_norm,
                   lambda_phys=1.0, train=True, dropout_key=None,
                   X_mean=None, X_std=None, Y_mean=None, Y_std=None,
                   colloc_L_norm=None):
    if train and dropout_key is not None:
        preds_norm = model.apply(params, x_norm, train=True, rngs={'dropout': dropout_key})
    else:
        preds_norm = model.apply(params, x_norm, train=False)

    preds_phys = preds_norm * Y_std + Y_mean
    y_phys     = y_norm * Y_std + Y_mean
    data_loss  = jnp.mean((preds_phys - y_phys) ** 2)

    physics_loss_data = 0.0
    colloc_loss = 0.0
    if lambda_phys > 0:
        L_phys = x_norm * X_std + X_mean
        T_phys = vec6_to_sym3(preds_phys)
        residuals_data = maxwellB_residual(L_phys.reshape(-1, 3, 3), T_phys)
        physics_loss_data = jnp.mean(residuals_data ** 2)
        if colloc_L_norm is not None:
            colloc_L_phys = colloc_L_norm * X_std + X_mean
            colloc_pred_norm = model.apply(params, colloc_L_norm, train=False)
            colloc_pred_phys = colloc_pred_norm * Y_std + Y_mean
            colloc_T_phys    = vec6_to_sym3(colloc_pred_phys)
            residuals_colloc = maxwellB_residual(colloc_L_phys.reshape(-1, 3, 3), colloc_T_phys)
            colloc_loss = jnp.mean(residuals_colloc ** 2)

    total_phys_loss = physics_loss_data + colloc_loss
    total_loss = data_loss + lambda_phys * total_phys_loss
    return total_loss, (data_loss, total_phys_loss)

# --- Train step ---
def make_train_step(model, optimizer, lambda_phys, X_mean, X_std, Y_mean, Y_std, colloc_L_norm):
    @jax.jit
    def train_step(params, opt_state, x, y, dropout_key):
        (loss_val, (d_loss, p_loss)), grads = jax.value_and_grad(compute_losses, has_aux=True)(
            params, model, x, y, lambda_phys, True, dropout_key,
            X_mean, X_std, Y_mean, Y_std,
            colloc_L_norm=colloc_L_norm
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val, d_loss, p_loss
    return train_step

# --- Cosine LR schedule ---
def cosine_annealing_lr(init_lr, T_max_epochs, steps_per_epoch):
    T_max_steps = T_max_epochs * steps_per_epoch
    def schedule_fn(step):
        return init_lr * 0.5 * (1 + jnp.cos(jnp.pi * step / T_max_steps))
    return schedule_fn

# --- Main training function ---
@hydra.main(config_path="config/train", config_name="maxwellBtensor_config", version_base=None)
def main(cfg: DictConfig):
    # Load and normalize dataset
    X_path = f"./datafiles/X_3D_{cfg.model_type}.pt"
    Y_path = f"./datafiles/Y_3D_{cfg.model_type}.pt"
    X_train, X_val, X_test, Y_train, Y_val, Y_test, X_mean, X_std, Y_mean, Y_std = \
        load_and_normalize_data(X_path, Y_path, seed=cfg.seed)

    X_train, X_val, X_test = jnp.array(X_train), jnp.array(X_val), jnp.array(X_test)
    Y_train, Y_val, Y_test = jnp.array(Y_train), jnp.array(Y_val), jnp.array(Y_test)

    # Init model
    activation_fn = activation_map.get(cfg.model.activation, nn.relu)
    model_layers = list(cfg.model.layers)
    model_layers[-1] = 6
    model = MLP(features=model_layers, dropout=cfg.model.dropout, activation_fn=activation_fn)

    key = jax.random.PRNGKey(cfg.seed)
    dummy_input = jnp.ones([1, X_train.shape[1]])
    params = model.init(key, dummy_input)

    # Collocation points for physics loss
    colloc_L_norm = None
    if cfg.training.lambda_phys > 0:
        key_colloc = jax.random.PRNGKey(cfg.seed + 123)
        colloc_L_norm = jax.random.uniform(
            key_colloc, shape=(cfg.training.n_colloc, X_train.shape[1]),
            minval=-3.0, maxval=3.0)

    # Optimizer
    steps_per_epoch = int(np.ceil(X_train.shape[0] / cfg.training.batch_size))
    lr_schedule_fn = cosine_annealing_lr(cfg.training.learning_rate, cfg.training.num_epochs, steps_per_epoch)
    optimizer = optax.adamw(learning_rate=lr_schedule_fn, weight_decay=cfg.training.weight_decay)
    opt_state = optimizer.init(params)

    # Train loop
    train_step = make_train_step(model, optimizer, cfg.training.lambda_phys, X_mean, X_std, Y_mean, Y_std, colloc_L_norm)
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(cfg.training.num_epochs):
        perm = np.random.permutation(X_train.shape[0])
        X_sh, Y_sh = X_train[perm], Y_train[perm]
        total_loss_ep, total_dloss, total_ploss = 0, 0, 0
        dropout_key = jax.random.fold_in(key, epoch)

        for i in range(steps_per_epoch):
            s, e = i * cfg.training.batch_size, min((i+1)*cfg.training.batch_size, X_train.shape[0])
            xb, yb = X_sh[s:e], Y_sh[s:e]
            params, opt_state, loss_val, d_loss, p_loss = train_step(params, opt_state, xb, yb, dropout_key)
            total_loss_ep += loss_val.item() * (e - s)
            total_dloss += d_loss.item() * (e - s)
            total_ploss += p_loss.item() * (e - s)

        avg_total_loss = total_loss_ep / X_train.shape[0]
        avg_data_loss = total_dloss / X_train.shape[0]
        avg_phys_loss = total_ploss / X_train.shape[0]
        train_losses.append(avg_total_loss)

        # Validation losses
        avg_val_loss, (val_data_loss, val_phys_loss) = compute_losses(
            params, model, X_val, Y_val, cfg.training.lambda_phys, train=False,
            X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std,
            colloc_L_norm=colloc_L_norm)
        avg_val_loss, val_data_loss, val_phys_loss = map(float, [avg_val_loss, val_data_loss, val_phys_loss])
        val_losses.append(avg_val_loss)

        # Early stopping (basic)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(params, X_mean, X_std, Y_mean, Y_std,
                            os.path.join(cfg.output_dir, f"pinn_{cfg.model_type}", "trained_params.msgpack"))

    # Plot curves
    fig_dir = os.path.join(cfg.output_dir, f"pinn_{cfg.model_type}", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_learning_curves(train_losses, val_losses, fig_dir, cfg.model_type)

    # --- Test evaluation ---
    ckpt_path = os.path.join(cfg.output_dir, f"pinn_{cfg.model_type}", "trained_params.msgpack")
    restored = load_checkpoint(ckpt_path, {"params": params, "X_mean": X_mean, "X_std": X_std,
                                           "Y_mean": Y_mean, "Y_std": Y_std})
    best_params = restored["params"]

    # Compute val metrics
    y_val_true_phys = np.array(Y_val) * np.array(Y_std) + np.array(Y_mean)
    y_val_pred_phys = np.array(model.apply(params, X_val, train=False)) * np.array(Y_std) + np.array(Y_mean)
    val_mse = float(np.mean((y_val_true_phys - y_val_pred_phys) ** 2))
    val_mae = mean_absolute_error(y_val_true_phys, y_val_pred_phys)

    # Compute test metrics
    test_total_loss, (test_data_loss_norm, test_phys_loss_norm) = compute_losses(
        best_params, model, X_test, Y_test, cfg.training.lambda_phys, train=False,
        X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std,
        colloc_L_norm=colloc_L_norm)
    test_total_loss = float(test_total_loss)
    test_data_loss_norm = float(test_data_loss_norm)
    test_phys_loss_norm = float(test_phys_loss_norm)

    y_true_phys = np.array(Y_test) * np.array(Y_std) + np.array(Y_mean)
    y_pred_phys = np.array(model.apply(best_params, X_test, train=False)) * np.array(Y_std) + np.array(Y_mean)
    test_mse_phys = float(np.mean((y_true_phys - y_pred_phys) ** 2))
    test_mae_phys = mean_absolute_error(y_true_phys, y_pred_phys)

    # Choose some random or fixed test samples to visualize
    sample_ids = [0, 5, 10]  # You can randomize this if you prefer
    plot_stress_tensor_comparison(vec6_to_sym3, y_true_phys, y_pred_phys, sample_ids, fig_dir, cfg.model_type)

    # Plot residuals
    residuals = y_true_phys - y_pred_phys
    plot_residual_hist(residuals, fig_dir, cfg.model_type)
    plot_residuals_vs_pred(y_pred_phys, residuals, fig_dir, cfg.model_type)

    # Dynamic metrics table
    metrics_table = []
    if cfg.training.lambda_phys > 0:
        metrics_table.extend([
            ["Train/total_loss", avg_total_loss],
            ["Train/data_loss", avg_data_loss],
            ["Train/physics_loss", avg_phys_loss],
            ["Val/total_loss", avg_val_loss],
            ["Val/data_loss", val_data_loss],
            ["Val/physics_loss", val_phys_loss],
            ["Val/MAE", val_mae],
            ["Val/MSE", val_mse],
            ["Test/total_loss", test_total_loss],
            ["Test/data_loss", test_data_loss_norm],
            ["Test/physics_loss", test_phys_loss_norm],
            ["Test/MAE", test_mae_phys],
            ["Test/MSE", test_mse_phys],
        ])
    else:
        metrics_table.extend([
            ["Train/total_loss", avg_total_loss],
            ["Train/data_loss", avg_data_loss],
            ["Val/total_loss", avg_val_loss],
            ["Val/data_loss", val_data_loss],
            ["Val/MAE", val_mae],
            ["Val/MSE", val_mse],
            ["Test/total_loss", test_total_loss],
            ["Test/data_loss", test_data_loss_norm],
            ["Test/MAE", test_mae_phys],
            ["Test/MSE", test_mse_phys],
        ])

    print("\n" + tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="grid"))

if __name__ == "__main__":
    main()




    import os
import numpy as np
import torch
import jax.numpy as jnp
import flax
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from scipy.stats import gaussian_kde


def load_and_normalize_data_maxwellB(X_path, Y_path, seed=42):
    """
    Maxwell-B data loader:
    - Per-component balanced split: ensures extremes of EACH stress component
      appear in train, val, and test sets.
    - Normalises using TRAIN statistics.
    - Prints debug min / max / std for Y physical units per split.
    """

    rng = np.random.default_rng(seed)

    # -------------------------------------------------
    # 1. Load dataset (Physical Units)
    # -------------------------------------------------
    X = torch.load(X_path).numpy()
    Y = torch.load(Y_path).numpy()

    n_samples = X.shape[0]
    n_outputs = Y.shape[1]  # should be 6 for 3D symmetric stress tensor

    # -------------------------------------------------
    # 2. Initialise split index sets
    # -------------------------------------------------
    idx_train_set = set()
    idx_val_set = set()
    idx_test_set = set()

    # -------------------------------------------------
    # 3. Per-component splitting
    # -------------------------------------------------
    for comp in range(n_outputs):
        # Sort indices by magnitude of this component
        sorted_idx = np.argsort(np.abs(Y[:, comp]))[::-1]  # descending

        # Pure round robin across full sorted list
        train_idx = sorted_idx[0::3]
        val_idx   = sorted_idx[1::3]
        test_idx  = sorted_idx[2::3]

        # Add to component-specific sets
        idx_train_set.update(train_idx)
        idx_val_set.update(val_idx)
        idx_test_set.update(test_idx)

    # -------------------------------------------------
    # 4. Convert sets to lists & shuffle for randomness
    # -------------------------------------------------
    idx_train = list(idx_train_set)
    idx_val   = list(idx_val_set)
    idx_test  = list(idx_test_set)

    rng.shuffle(idx_train)
    rng.shuffle(idx_val)
    rng.shuffle(idx_test)

    # -------------------------------------------------
    # 5. Create splits
    # -------------------------------------------------
    X_train, Y_train = X[idx_train], Y[idx_train]
    X_val,   Y_val   = X[idx_val],   Y[idx_val]
    X_test,  Y_test  = X[idx_test],  Y[idx_test]

    # -------------------------------------------------
    # 6. Normalisation using TRAIN stats
    # -------------------------------------------------
    X_mean = X_train.mean(axis=0)
    X_std  = X_train.std(axis=0)
    X_std[X_std == 0] = 1

    Y_mean = Y_train.mean(axis=0)
    Y_std  = Y_train.std(axis=0)
    Y_std[Y_std == 0] = 1

    X_train_n = (X_train - X_mean) / X_std
    X_val_n   = (X_val   - X_mean) / X_std
    X_test_n  = (X_test  - X_mean) / X_std

    Y_train_n = (Y_train - Y_mean) / Y_std
    Y_val_n   = (Y_val   - Y_mean) / Y_std
    Y_test_n  = (Y_test  - Y_mean) / Y_std

    # -------------------------------------------------
    # 7. Debug: Check Y ranges in physical units
    # -------------------------------------------------
    def print_stats(name, arr):
        print(f"{name}  Y (Physical Units):")
        print("  Min :", arr.min(axis=0))
        print("  Max :", arr.max(axis=0))
        print("  Std :", arr.std(axis=0))

    print("\n=== Normalisation Sanity Check (MaxwellB Per-Component Balanced Split) ===")
    print_stats("Train", Y_train)
    print_stats("Val  ", Y_val)
    print_stats("Test ", Y_test)

    return (jnp.array(X_train_n), jnp.array(X_val_n), jnp.array(X_test_n),
            jnp.array(Y_train_n), jnp.array(Y_val_n), jnp.array(Y_test_n),
            X_mean, X_std, Y_mean, Y_std)


# =============================================================================
# Model checkpoint utilities
# =============================================================================

def save_checkpoint(params, X_mean, X_std, Y_mean, Y_std, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    to_save = {
        "params": params,
        "X_mean": X_mean,
        "X_std": X_std,
        "Y_mean": Y_mean,
        "Y_std": Y_std
    }
    with open(path, "wb") as f:
        f.write(flax.serialization.to_bytes(to_save))

def load_checkpoint(path, init_params):
    with open(path, "rb") as f:
        restored = flax.serialization.from_bytes(init_params, f.read())
    return restored


# =============================================================================
# Plotting utilities
# =============================================================================

def plot_learning_curves(train_losses, val_losses, fig_dir, model_type):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,  label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Learning Curves ({model_type})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "learning_curves.png"))
    plt.close()

def plot_residual_hist(residuals, fig_dir, model_type):
    residuals_1d = np.ravel(residuals)
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals_1d, bins=30, color='skyblue', stat='density')
    kde = gaussian_kde(residuals_1d)
    x_range = np.linspace(residuals_1d.min(), residuals_1d.max(), 1000)
    plt.plot(x_range, kde(x_range), color='orange', lw=2, label='KDE')
    plt.axvline(0, color='red', linestyle='--', label='Zero Residual')
    plt.title(f"Residuals on Test Data ({model_type})")
    plt.xlabel("Residual (True - Predicted)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "residual_histogram.png"))
    plt.close()

def plot_residuals_vs_pred(y_pred, residuals, fig_dir, model_type):
    y_pred_1d = np.ravel(y_pred)
    residuals_1d = np.ravel(residuals)
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred_1d, residuals_1d, alpha=0.5)
    smoothed = sm.nonparametric.lowess(residuals_1d, y_pred_1d, frac=0.3)
    plt.plot(smoothed[:, 0], smoothed[:, 1], color='red', lw=2, label='LOWESS')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residual (True - Predicted)")
    plt.title(f"Residuals vs Predicted ({model_type})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "residuals_vs_predictions.png"))
    plt.close()

    
def plot_stress_tensor_comparison(vec6_to_sym3, y_true_phys, y_pred_phys, sample_indices, save_dir, model_type):
    """
    Plot true vs predicted symmetric 3x3 stress tensors for given sample indices.
    Saves each comparison as a side-by-side heatmap.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for idx in sample_indices:
        # Convert Voigt 6-vector to 3x3 full tensor
        T_true = np.array(vec6_to_sym3(jnp.array([y_true_phys[idx]]))).squeeze()
        T_pred = np.array(vec6_to_sym3(jnp.array([y_pred_phys[idx]]))).squeeze()
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # True tensor heatmap
        im0 = axes[0].imshow(T_true, cmap="viridis")
        for (i, j), val in np.ndenumerate(T_true):
            axes[0].text(j, i, f"{val:.2e}", ha='center', va='center', color='white')
        axes[0].set_title(f"True Stress Tensor (sample {idx})")
        fig.colorbar(im0, ax=axes[0], format="%.0e")
        
        # Predicted tensor heatmap
        im1 = axes[1].imshow(T_pred, cmap="viridis")
        for (i, j), val in np.ndenumerate(T_pred):
            axes[1].text(j, i, f"{val:.2e}", ha='center', va='center', color='white')
        axes[1].set_title(f"Predicted Stress Tensor (sample {idx})")
        fig.colorbar(im1, ax=axes[1], format="%.0e")
        
        plt.suptitle(f"{model_type} Stress Tensor Comparison")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"stress_tensor_comparison_sample_{idx}.png"))
        plt.close()