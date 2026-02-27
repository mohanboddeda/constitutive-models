# ==============================
# Imports
# ==============================
import os
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from omegaconf import DictConfig
import hydra
from tabulate import tabulate
from sklearn.metrics import mean_absolute_error

# Import from new dimensionless utils file
from utils.data_utils_maxwellBdim import (
    load_and_normalize_data_maxwellB_dimless,
    save_checkpoint,
    load_checkpoint,
    plot_learning_curves_physical,
    plot_residual_hist,
    plot_residuals_vs_pred,
    plot_stress_tensor_comparison
)

# ==============================
# Maxwell-B physics constants
# ==============================
ETA0 = 5.28e-5  # Pa·s
LAM = 1.902     # s

# ============================================================
# Helper: vector -> symmetric tensor
# ============================================================
def vec6_to_sym3(vec):
    T = jnp.zeros((vec.shape[0], 3, 3))
    T = T.at[:, 0, 0].set(vec[:, 0])
    T = T.at[:, 1, 1].set(vec[:, 1])
    T = T.at[:, 2, 2].set(vec[:, 2])
    T = T.at[:, 0, 1].set(vec[:, 3]); T = T.at[:, 1, 0].set(vec[:, 3])
    T = T.at[:, 0, 2].set(vec[:, 4]); T = T.at[:, 2, 0].set(vec[:, 4])
    T = T.at[:, 1, 2].set(vec[:, 5]); T = T.at[:, 2, 1].set(vec[:, 5])
    return T

# ============================================================
# Residuals
# ============================================================
def maxwellB_residual_dimless(L_hat, T_hat, Wi):
    """
    Dimensionless Maxwell-B steady-state residual:
    T_hat − Wi(L_hat^T T_hat + T_hat L_hat) − 2 D_hat = 0
    """
    D_hat = 0.5 * (L_hat + jnp.swapaxes(L_hat, 1, 2))
    LTt = jnp.matmul(jnp.swapaxes(L_hat, 1, 2), T_hat)
    TL  = jnp.matmul(T_hat, L_hat)
    return T_hat - Wi * (LTt + TL) - 2.0 * D_hat

def maxwellB_residual(L_phys, T_phys):
    """
    Physical-units Maxwell-B steady-state residual:
    T − λ(L^T T + T L) − 2 η0 D = 0
    """
    D = 0.5 * (L_phys + jnp.swapaxes(L_phys, 1, 2))
    LTt = jnp.matmul(jnp.swapaxes(L_phys, 1, 2), T_phys)
    TL  = jnp.matmul(T_phys, L_phys)
    return T_phys - LAM * (LTt + TL) - 2.0 * ETA0 * D

# ============================================================
# Activation map
# ============================================================
activation_map = {
    "relu": nn.relu,
    "tanh": nn.tanh,
    "sigmoid": nn.sigmoid
}

# ============================================================
# MLP
# ============================================================
class MLP(nn.Module):
    features: list
    dropout: float = 0.0
    activation_fn: callable = None

    @nn.compact
    def __call__(self, x, train=True):
        if x.ndim == 3:
            x = x.reshape((x.shape[0], -1))
        act_fn = self.activation_fn or nn.relu
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = act_fn(x)
            x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
        return nn.Dense(self.features[-1])(x)

# ============================================================
# Compute Losses
# ============================================================
def compute_losses(params, model, x_norm, y_norm, lambda_phys,
                   train, dropout_key,
                   X_mean, X_std, Y_mean, Y_std,
                   scaling_mode, Wi=None, sigma_ref=None):

    if train and dropout_key is not None:
        preds_norm = model.apply(params, x_norm, train=True, rngs={'dropout': dropout_key})
    else:
        preds_norm = model.apply(params, x_norm, train=False)

    if scaling_mode == "dimensionless":
        preds_T_hat = preds_norm * Y_std + Y_mean
        y_T_hat     = y_norm   * Y_std + Y_mean
        data_loss = jnp.mean((preds_T_hat - y_T_hat) ** 2)

        physics_loss_data = 0.0
        if lambda_phys > 0:
            L_hat = x_norm.reshape(-1, 3, 3)
            T_hat = vec6_to_sym3(preds_T_hat)
            residuals_data = maxwellB_residual_dimless(L_hat, T_hat, Wi)
            physics_loss_data = jnp.mean(residuals_data ** 2)

        total_loss = data_loss + lambda_phys * physics_loss_data
        return total_loss, (data_loss, physics_loss_data)

    elif scaling_mode in ["standard", "minmax"]:
        preds_phys = preds_norm * Y_std + Y_mean
        y_phys = y_norm * Y_std + Y_mean
        data_loss = jnp.mean((preds_phys - y_phys) ** 2)

        physics_loss_data = 0.0
        if lambda_phys > 0:
            L_phys = x_norm * X_std + X_mean
            T_phys = vec6_to_sym3(preds_phys)
            residuals_data = maxwellB_residual(L_phys.reshape(-1, 3, 3), T_phys)
            physics_loss_data = jnp.mean(residuals_data ** 2)

        total_loss = data_loss + lambda_phys * physics_loss_data
        return total_loss, (data_loss, physics_loss_data)

    else:
        raise ValueError(f"Unknown scaling_mode: {scaling_mode}")

# ============================================================
# Train step
# ============================================================
def make_train_step(model, optimizer, lambda_phys, X_mean, X_std,
                    Y_mean, Y_std, scaling_mode, Wi=None, sigma_ref=None):
    @jax.jit
    def train_step(params, opt_state, x, y, dropout_key):
        (loss_val, (d_loss, p_loss)), grads = jax.value_and_grad(compute_losses, has_aux=True)(
            params, model, x, y,
            lambda_phys, True, dropout_key,
            X_mean, X_std, Y_mean, Y_std,
            scaling_mode, Wi, sigma_ref
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val, d_loss, p_loss
    return train_step

# ============================================================
# Cosine LR
# ============================================================
def cosine_annealing_lr(init_lr, T_max_epochs, steps_per_epoch):
    T_max_steps = T_max_epochs * steps_per_epoch
    def schedule_fn(step):
        return init_lr * 0.5 * (1 + jnp.cos(jnp.pi * step / T_max_steps))
    return schedule_fn

# ============================================================
# Main training
# ============================================================
def run_training_for_lambda(cfg, lambda_val):
    out_dir = os.path.join(cfg.output_dir, f"pinn_{cfg.model_type}", "lambda_sweep", f"lambda_{lambda_val}")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Load data
    X_train, X_val, X_test, Y_train, Y_val, Y_test, X_mean, X_std, Y_mean, Y_std, extras = \
        load_and_normalize_data_maxwellB_dimless(
            f"./datafiles/X_3D_{cfg.model_type}.pt",
            f"./datafiles/Y_3D_{cfg.model_type}.pt",
            seed=cfg.seed,
            balanced_split=cfg.data.balanced_split
        )
    Wi = extras["Wi"]
    sigma_ref = extras["sigma_ref_frob"]

    # Convert to JAX arrays
    X_train, X_val, X_test = jnp.array(X_train), jnp.array(X_val), jnp.array(X_test)
    Y_train, Y_val, Y_test = jnp.array(Y_train), jnp.array(Y_val), jnp.array(Y_test)

    activation_fn = activation_map.get(cfg.model.activation, nn.relu)
    model_layers = list(cfg.model.layers)
    model_layers[-1] = 6
    model = MLP(features=model_layers, dropout=cfg.model.dropout, activation_fn=activation_fn)

    key = jax.random.PRNGKey(cfg.seed)
    params = model.init(key, jnp.ones([1, X_train.shape[1]]))

    steps_per_epoch = int(np.ceil(X_train.shape[0] / cfg.training.batch_size))
    lr_schedule_fn = cosine_annealing_lr(cfg.training.learning_rate, cfg.training.num_epochs, steps_per_epoch)
    optimizer = optax.adamw(learning_rate=lr_schedule_fn, weight_decay=cfg.training.weight_decay)
    opt_state = optimizer.init(params)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    # Warmup: no physics in first 40% epochs
    warmup_epochs = int(0.4 * cfg.training.num_epochs)

    for epoch in range(cfg.training.num_epochs):
        if epoch < warmup_epochs:
            lambda_curr = 0.0
        else:
            frac = (epoch - warmup_epochs) / (cfg.training.num_epochs - warmup_epochs)
            lambda_curr = lambda_val * frac

        train_step = make_train_step(model, optimizer, lambda_curr, X_mean, X_std,
                                     Y_mean, Y_std, cfg.data.scaling_mode, Wi, sigma_ref)

        perm = np.random.permutation(X_train.shape[0])
        X_sh, Y_sh = X_train[perm], Y_train[perm]
        total_loss_ep, total_dloss, total_ploss = 0, 0, 0
        dropout_key = jax.random.fold_in(key, epoch)

        for i in range(steps_per_epoch):
            s, e = i * cfg.training.batch_size, min((i+1) * cfg.training.batch_size, X_train.shape[0])
            xb, yb = X_sh[s:e], Y_sh[s:e]
            params, opt_state, loss_val, d_loss, p_loss = train_step(params, opt_state, xb, yb, dropout_key)
            total_loss_ep += loss_val.item() * (e - s)
            total_dloss += d_loss.item() * (e - s)
            total_ploss += p_loss.item() * (e - s)

        avg_total_loss = total_loss_ep / X_train.shape[0]
        avg_data_loss = total_dloss / X_train.shape[0]
        avg_phys_loss = total_ploss / X_train.shape[0]
        train_losses.append(avg_total_loss)

        avg_val_loss, (val_data_loss, val_phys_loss) = compute_losses(
            params, model, X_val, Y_val,
            lambda_curr, False, None,
            X_mean, X_std, Y_mean, Y_std,
            cfg.data.scaling_mode, Wi, sigma_ref
        )
        avg_val_loss, val_data_loss, val_phys_loss = map(float, [avg_val_loss, val_data_loss, val_phys_loss])
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(params, X_mean, X_std, Y_mean, Y_std, os.path.join(out_dir, "trained_params.msgpack"))

        if epoch % 50 == 0 or epoch == cfg.training.num_epochs - 1:
            print(f"Epoch {epoch}: λ_phys = {lambda_curr:.4f}, Val Loss = {avg_val_loss:.6e}")

    # Plot learning curves
    plot_learning_curves_physical(train_losses, val_losses, sigma_ref, fig_dir, cfg.model_type)

    # Restore best parameters
    ckpt_path = os.path.join(out_dir, "trained_params.msgpack")
    restored = load_checkpoint(ckpt_path, {"params": params, "X_mean": X_mean, "X_std": X_std,
                                           "Y_mean": Y_mean, "Y_std": Y_std})
    best_params = restored["params"]

    def de_normalize_dimless(Y_norm):
        T_hat = np.array(Y_norm) * np.array(Y_std) + np.array(Y_mean)
        return T_hat * sigma_ref

    # Evaluate metrics
    y_true_phys = de_normalize_dimless(Y_test)
    y_pred_phys = de_normalize_dimless(model.apply(best_params, X_test, train=False))

    plot_stress_tensor_comparison(vec6_to_sym3, y_true_phys, y_pred_phys, [0, 5, 10], fig_dir, cfg.model_type)
    residuals = y_true_phys - y_pred_phys
    plot_residual_hist(residuals, fig_dir, cfg.model_type)
    plot_residuals_vs_pred(y_pred_phys, residuals, fig_dir, cfg.model_type)

    val_mse = float(np.mean((de_normalize_dimless(Y_val) - de_normalize_dimless(model.apply(best_params, X_val, train=False))) ** 2))
    val_mae = mean_absolute_error(de_normalize_dimless(Y_val),
                                  de_normalize_dimless(model.apply(best_params, X_val, train=False)))

    test_total_loss, (test_data_loss_norm, test_phys_loss_norm) = compute_losses(
        best_params, model, X_test, Y_test,
        lambda_val, False, None,
        X_mean, X_std, Y_mean, Y_std,
        cfg.data.scaling_mode, Wi, sigma_ref
    )
    test_total_loss = float(test_total_loss)
    test_data_loss_norm = float(test_data_loss_norm)
    test_phys_loss_norm = float(test_phys_loss_norm)
    test_mse_phys = float(np.mean((y_true_phys - y_pred_phys) ** 2))
    test_mae_phys = mean_absolute_error(y_true_phys, y_pred_phys)

    metrics_table = [
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
    ]
    print(f"\n=== Metrics for λ_phys = {lambda_val} ===")
    print(tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="grid"))

    return [lambda_val, val_mae, val_mse, val_phys_loss, test_mae_phys, test_mse_phys, test_phys_loss_norm]

# ============================================================
@hydra.main(config_path="config/train", config_name="maxwellBtensor_config1", version_base=None)
def main(cfg: DictConfig):
    results_summary = []
    print("\n=== Dynamic λ_phys Training ===")
    for target_lambda_phys in cfg.training.lambda_phys:
        print(f"\n--- Training to λ_phys = {target_lambda_phys} ---")
        results_summary.append(run_training_for_lambda(cfg, target_lambda_phys))

    print("\n=== Summary ===")
    print(tabulate(results_summary,
                   headers=["Target λ_phys", "Val_MAE", "Val_MSE", "Val_phys_loss",
                            "Test_MAE", "Test_MSE", "Test_phys_loss"],
                   tablefmt="grid"))

if __name__ == "__main__":
    main()