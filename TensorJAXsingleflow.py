# ================================================================
# Imports
# ================================================================
import os, time
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from sklearn.metrics import mean_absolute_error
from tabulate import tabulate
import GPUtil
from omegaconf import DictConfig
import hydra

# ================================================================
# Local utilities
# ================================================================
# ⚠️ Make sure save_checkpoint / load_checkpoint come from datautilsflow.py
# and not from somewhere else
from utils.datautilsflow import (
    save_checkpoint,
    load_checkpoint
)

# Plotting helpers
from utils.data_utils_stable import (
    plot_all_losses,
    plot_residual_hist,
    plot_residuals_vs_pred,
    plot_stress_tensor_comparison
)

# ================================================================
# Helper: Convert 6-component symmetric vector to full 3x3 matrix
# ================================================================
def vec6_to_sym3(vec):
    """
    Converts a batch of 6-component symmetric tensor vectors to 3×3 tensors.
    Order: [Txx, Tyy, Tzz, Txy, Txz, Tyz]
    """
    T = jnp.zeros((vec.shape[0], 3, 3))
    T = T.at[:, 0, 0].set(vec[:, 0])
    T = T.at[:, 1, 1].set(vec[:, 1])
    T = T.at[:, 2, 2].set(vec[:, 2])
    T = T.at[:, 0, 1].set(vec[:, 3]); T = T.at[:, 1, 0].set(vec[:, 3])
    T = T.at[:, 0, 2].set(vec[:, 4]); T = T.at[:, 2, 0].set(vec[:, 4])
    T = T.at[:, 1, 2].set(vec[:, 5]); T = T.at[:, 2, 1].set(vec[:, 5])
    return T

# ================================================================
# Maxwell-B Residual Function
# ================================================================
def maxwellB_residual(L_phys, T_phys, eta0, lam):
    """
    Constitutive residual:
    R = (I - lambda L) T + T (-lambda L^T) - 2 * eta0 * D
    """
    D = 0.5 * (L_phys + jnp.swapaxes(L_phys, 1, 2))  # symmetric part
    dim = L_phys.shape[1]
    I = jnp.eye(dim)
    A = I - lam * L_phys
    B = -lam * jnp.swapaxes(L_phys, 1, 2)
    C = 2.0 * eta0 * D
    R = jnp.matmul(A, T_phys) + jnp.matmul(T_phys, B) - C
    return R

# ================================================================
# Activation map for config selection
# ================================================================
def sine(x):
    return jnp.sin(10.0 * x)

activation_map = {
    "relu": nn.relu,
    "tanh": nn.tanh,
    "sigmoid": nn.sigmoid,
    "sine": sine
}

# ================================================================
# MLP Network Definition
# ================================================================
class MLP(nn.Module):
    features: list
    dropout: float = 0.0
    activation_fn: callable = None

    @nn.compact
    def __call__(self, x, train=True):
        """
        Forward pass for MLP.
        Flattens (batch, 3, 3) to (batch, 9) if needed.
        """
        if x.ndim == 3:
            x = x.reshape((x.shape[0], -1))
        # Always set the activation function
        act_fn = self.activation_fn or nn.relu

        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = act_fn(x)  # ✅ apply the activation function
            x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
        return nn.Dense(self.features[-1])(x)  # last layer outputs 6 values

# ================================================================
# Loss function: Data loss + Physics loss
# ================================================================
def compute_losses(params, model, x_norm, y_norm,
                   lambda_phys, train, dropout_key,
                   X_mean, X_std, Y_mean, Y_std,
                   residual_fn, eta0, lam):
    """
    Compute total loss and individual components.
    Both data loss and physics loss are computed in physical units.

    x_norm, y_norm: normalized input/output data
    X_mean, X_std, Y_mean, Y_std: normalization statistics
    """

    # Forward pass
    use_dropout = (train and (dropout_key is not None))
    preds_norm = model.apply(params, x_norm, train=train,
                             rngs={'dropout': dropout_key} if use_dropout else {})

    # 1. Data Loss (calculated in physical units and MSE Loss)
    # De-normalize to physical units
    preds_phys = preds_norm * Y_std + Y_mean
    y_phys     = y_norm    * Y_std + Y_mean
    # Data loss (MSE in physical units)
    data_loss = jnp.mean((preds_phys - y_phys) ** 2)

    # Physics loss (MSE of residuals in physical units)
    L_phys = x_norm * X_std + X_mean
    T_phys = vec6_to_sym3(preds_phys)
    residuals_data = residual_fn(L_phys.reshape(-1, 3, 3), T_phys, eta0, lam)
    physics_loss_data = jnp.mean(residuals_data ** 2)

    # Combine losses
    scaled_physics_loss = jnp.where(lambda_phys > 0.0,
                                    lambda_phys * physics_loss_data,
                                    0.0)
    total_loss = data_loss + scaled_physics_loss

    return total_loss, (data_loss, physics_loss_data)

# ================================================================
# JIT-compiled training step
# ================================================================
def make_train_step(model, optimizer,
                    X_mean, X_std, Y_mean, Y_std,
                    residual_fn, eta0, lam):
    @jax.jit
    def train_step(params, opt_state, x, y, lambda_phys_curr, dropout_key):
        (loss_val, (d_loss, p_loss)), grads = jax.value_and_grad(
            compute_losses, has_aux=True
        )(params, model, x, y, lambda_phys_curr, True, dropout_key,
          X_mean, X_std, Y_mean, Y_std, residual_fn, eta0, lam)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val, d_loss, p_loss
    return train_step

# ================================================================
# Cosine Annealing LR Schedule
# ================================================================
def cosine_annealing_lr(init_lr, T_max_epochs, steps_per_epoch):
    T_max_steps = T_max_epochs * steps_per_epoch
    def schedule_fn(step):
        return init_lr * 0.5 * (1 + jnp.cos(jnp.pi * step / T_max_steps))
    return schedule_fn

# ================================================================
# Main training function for a single flow type
# ================================================================
def train_maxwell_singleflow_with_hydra(cfg: DictConfig):
    """
    Trains Maxwell-B PINN for one flow type specified in cfg.
    """

    # --- Load config values ---
    flow_type = cfg.flow_type
    eta0 = cfg.rheology.eta0
    lam = cfg.rheology.lam
    lambda_phys_target = cfg.training.lambda_phys
    num_epochs = cfg.training.num_epochs
    batch_size = cfg.training.batch_size
    learning_rate = cfg.training.learning_rate
    weight_decay = cfg.training.weight_decay
    layers = cfg.model.layers
    dropout = cfg.model.dropout
    activation = cfg.model.activation

    # --- Directories ---
    start_time = time.time()
    folder_path = os.path.join(cfg.output_dir, flow_type)
    os.makedirs(folder_path, exist_ok=True)
    fig_dir = os.path.join(folder_path, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # --- Load preprocessed normalized data ---
    try:
        X_train = np.load(os.path.join(folder_path, "X_train.npy"))
        X_val   = np.load(os.path.join(folder_path, "X_val.npy"))
        X_test  = np.load(os.path.join(folder_path, "X_test.npy"))
        Y_train = np.load(os.path.join(folder_path, "Y_train.npy"))
        Y_val   = np.load(os.path.join(folder_path, "Y_val.npy"))
        Y_test  = np.load(os.path.join(folder_path, "Y_test.npy"))
        X_mean  = np.load(os.path.join(folder_path, "X_mean.npy"))
        X_std   = np.load(os.path.join(folder_path, "X_std.npy"))
        Y_mean  = np.load(os.path.join(folder_path, "Y_mean.npy"))
        Y_std   = np.load(os.path.join(folder_path, "Y_std.npy"))
    except FileNotFoundError as e:
        print(f"❌ Missing file: {e.filename}")
        print(f"Run datautilsflow.py to preprocess and save .npy files before training.")
        return

    print(f"\n📂 Training Maxwell-B for {flow_type}")
    print(f"Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")

    # --- Model init ---
    activation_fn = activation_map.get(activation, nn.tanh)
    model_layers = list(layers)
    model_layers[-1] = 6  # output size fixed at 6 for symmetric tensor
    model = MLP(features=model_layers, dropout=dropout, activation_fn=activation_fn)
    key = jax.random.PRNGKey(cfg.seed)
    params = model.init(key, jnp.ones([1, X_train.shape[1]]))

    # --- Optimizer ---
    steps_per_epoch = int(np.ceil(X_train.shape[0] / batch_size))
    lr_schedule_fn = cosine_annealing_lr(learning_rate, num_epochs, steps_per_epoch)
    optimizer = optax.adamw(learning_rate=lr_schedule_fn, weight_decay=weight_decay)
    opt_state = optimizer.init(params)

    # --- Loss tracking ---
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_data_losses, val_data_losses = [], []
    train_phys_losses, val_phys_losses = [], []
    residual_fn = maxwellB_residual

    # --- Training step ---
    train_step = make_train_step(model, optimizer,
                                 X_mean, X_std, Y_mean, Y_std,
                                 residual_fn, eta0, lam)

    # --- Main Loop ---
    for epoch in range(num_epochs):
        #lambda_curr = lambda_phys_target * (epoch / num_epochs) if num_epochs > 0 else lambda_phys_target
        lambda_curr = lambda_phys_target
        perm = np.random.permutation(X_train.shape[0])
        X_sh, Y_sh = X_train[perm], Y_train[perm]
        total_loss_ep, total_dloss, total_ploss = 0, 0, 0
        dropout_key = jax.random.fold_in(key, epoch)

        # Mini-batches
        for i in range(steps_per_epoch):
            s, e = i * batch_size, min((i+1) * batch_size, X_train.shape[0])
            xb, yb = X_sh[s:e], Y_sh[s:e]
            params, opt_state, loss_val, d_loss, p_loss = train_step(params, opt_state, xb, yb, lambda_curr, dropout_key)
            total_loss_ep += loss_val.item() * (e - s)
            total_dloss += d_loss.item() * (e - s)
            total_ploss += p_loss.item() * (e - s)

        # Epoch averages
        avg_total_loss = total_loss_ep / X_train.shape[0]
        avg_data_loss = total_dloss / X_train.shape[0]
        avg_phys_loss = total_ploss / X_train.shape[0]
        train_losses.append(avg_total_loss)
        train_data_losses.append(avg_data_loss)
        train_phys_losses.append(avg_phys_loss)

        # Validation
        avg_val_loss, (val_data_loss, val_phys_loss) = compute_losses(
            params, model, X_val, Y_val,
            lambda_curr, False, None,
            X_mean, X_std, Y_mean, Y_std,
            residual_fn, eta0, lam
        )
        avg_val_loss = float(avg_val_loss)
        val_losses.append(avg_val_loss)
        val_data_losses.append(float(val_data_loss))
        val_phys_losses.append(float(val_phys_loss))

        # Save checkpoint if val improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(params, X_mean, X_std, Y_mean, Y_std,
                            os.path.join(folder_path, f"{flow_type}_maxwellB.msgpack"))

        if epoch % 50 == 0 or epoch == num_epochs - 1:
            print(f"[{flow_type}] Epoch {epoch}: λ_phys={lambda_curr:.4f}, Val Loss={avg_val_loss:.6e}")

    # --- Post Training ---
    restored = load_checkpoint(os.path.join(folder_path, f"{flow_type}_maxwellB.msgpack"),
                               {"params": params, "X_mean": X_mean, "X_std": X_std,
                                "Y_mean": Y_mean, "Y_std": Y_std})
    best_params = restored["params"]

    # De-normalize helper
    def de_normalize(Y_norm):
        return np.array(Y_norm) * np.array(Y_std) + np.array(Y_mean)

    y_true_phys = de_normalize(Y_test)
    y_pred_phys = de_normalize(model.apply(best_params, X_test, train=False))

    # Plots
    plot_all_losses(train_losses, val_losses,
                    train_data_losses, val_data_losses,
                    train_phys_losses, val_phys_losses,
                    Y_std, fig_dir, "maxwell_B")
    plot_stress_tensor_comparison(vec6_to_sym3, y_true_phys, y_pred_phys, [0, 5, 10], fig_dir, "maxwell_B")
    residuals = y_true_phys - y_pred_phys
    plot_residual_hist(residuals, fig_dir, "maxwell_B")
    plot_residuals_vs_pred(y_pred_phys, residuals, fig_dir, "maxwell_B")

    # Final test metrics
    test_total_loss, (test_data_loss_norm, test_phys_loss_norm) = compute_losses(
        best_params, model, X_test, Y_test,
        lambda_phys_target, False, None,
        X_mean, X_std, Y_mean, Y_std,
        residual_fn, eta0, lam
    )
    test_mse_phys = float(np.mean((y_true_phys - y_pred_phys) ** 2))
    metrics_table = [
        ["Train/total_loss", avg_total_loss],
        ["Train/data_loss", avg_data_loss],
        ["Train/physics_loss", avg_phys_loss],
        ["Val/total_loss", best_val_loss],
        ["Val/data_loss", val_data_losses[-1]],
        ["Val/physics_loss", val_phys_losses[-1]],
        ["Test/total_loss", float(test_total_loss)],
        ["Test/data_loss", float(test_data_loss_norm)],
        ["Test/physics_loss", float(test_phys_loss_norm)],
        ["Test/MSE", test_mse_phys],
        ["Test/MAE", mean_absolute_error(y_true_phys, y_pred_phys)]
    ]
    print("\n=== Metrics ===")
    print(tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="grid"))

    elapsed_time = time.time() - start_time
    gpus = GPUtil.getGPUs()
    print(f"⏱ Training took {elapsed_time:.2f} s on {'GPU ' + gpus[0].name if gpus else 'CPU'}")
    return metrics_table

# ================================================================
# Hydra Entry Point
# ================================================================
@hydra.main(config_path="config/data", config_name="maxwell_config", version_base=None)
def main(cfg: DictConfig):
    if isinstance(cfg.flow_type, str):
        train_maxwell_singleflow_with_hydra(cfg)
    elif isinstance(cfg.flow_type, list):
        for ft in cfg.flow_type:
            temp_cfg = cfg.copy()
            temp_cfg.flow_type = ft
            train_maxwell_singleflow_with_hydra(temp_cfg)

if __name__ == "__main__":
    main()