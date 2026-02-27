import os
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
from omegaconf import DictConfig
import hydra
from sklearn.metrics import r2_score, mean_absolute_error
from torch.utils.tensorboard import SummaryWriter
from tabulate import tabulate

from utils.train_utils import (
    load_and_normalize_data,
    save_checkpoint,
    load_checkpoint,
    plot_learning_curves,
    plot_residual_hist,
    plot_residuals_vs_pred
)

# --- Maxwell-B constants ---
ETA0 = 5.28e-5
LAM = 1.902

# --- Tensor↔Voigt conversion ---
def vec6_to_sym3(vec):
    T = jnp.zeros((vec.shape[0], 3, 3))
    T = T.at[:,0,0].set(vec[:,0])
    T = T.at[:,1,1].set(vec[:,1])
    T = T.at[:,2,2].set(vec[:,2])
    T = T.at[:,0,1].set(vec[:,3]); T = T.at[:,1,0].set(vec[:,3])
    T = T.at[:,0,2].set(vec[:,4]); T = T.at[:,2,0].set(vec[:,4])
    T = T.at[:,1,2].set(vec[:,5]); T = T.at[:,2,1].set(vec[:,5])
    return T

def sym3_to_vec6(T):
    return jnp.stack([
        T[:,0,0],
        T[:,1,1],
        T[:,2,2],
        T[:,0,1],
        T[:,0,2],
        T[:,1,2]
    ], axis=1)

# Maxwell‑B residual
def maxwellB_residual(L_flat, T_pred_flat):
    L = L_flat.reshape(-1, 3, 3)
    T = vec6_to_sym3(T_pred_flat)
    D = 0.5 * (L + jnp.swapaxes(L, 1, 2))
    LTt = jnp.matmul(jnp.swapaxes(L, 1, 2), T)  # L^T T
    TL  = jnp.matmul(T, L)
    return T - LAM * (LTt + TL) - 2.0 * ETA0 * D

# --- Model ---
class MLP(nn.Module):
    features: list
    dropout: float = 0.0
    activation_fn: callable = nn.sigmoid

    @nn.compact
    def __call__(self, x, train=True):
        if x.ndim == 3:
            x = x.reshape((x.shape[0], -1))
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = self.activation_fn(x)
            if self.dropout > 0:
                x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
        return nn.Dense(self.features[-1])(x)

# --- Loss ---
def compute_losses(params, model, x, y, lambda_phys=1.0,
                   train=True, dropout_key=None,
                   X_mean=None, X_std=None, Y_mean=None, Y_std=None):

    if train and dropout_key is not None:
        preds = model.apply(params, x, train=True, rngs={'dropout': dropout_key})
    else:
        preds = model.apply(params, x, train=False)

    # normalized space data loss
    data_loss = jnp.mean((preds - y) ** 2)

    # normalized physics loss
    x_phys = x * X_std + X_mean
    preds_phys = preds * Y_std + Y_mean
    residuals_phys = maxwellB_residual(x_phys, preds_phys)
    residuals_voigt = sym3_to_vec6(residuals_phys)
    residuals_norm = residuals_voigt / Y_std
    physics_loss = jnp.mean(residuals_norm ** 2)

    total_loss = data_loss + lambda_phys * physics_loss
    return total_loss, (data_loss, physics_loss)

# --- Train step ---
def make_train_step(model, optimizer, lambda_phys, X_mean, X_std, Y_mean, Y_std):
    @jax.jit
    def train_step(params, opt_state, x, y, dropout_key):
        (loss_val, (d_loss, p_loss)), grads = jax.value_and_grad(compute_losses, has_aux=True)(
            params, model, x, y,
            lambda_phys,
            True, dropout_key,
            X_mean, X_std, Y_mean, Y_std
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val, d_loss, p_loss
    return train_step

@hydra.main(config_path="config/train", config_name="maxwellB_config", version_base=None)
def main(cfg: DictConfig):
    jax.config.update("jax_enable_x64", True)

    # --- Load and normalize data ---
    X_path = f"./datafiles/X_3D_{cfg.model_type}.pt"
    Y_path = f"./datafiles/Y_3D_{cfg.model_type}.pt"
    X_train, X_val, X_test, Y_train, Y_val, Y_test, X_mean, X_std, Y_mean, Y_std = \
        load_and_normalize_data(X_path, Y_path, seed=cfg.seed)

    # Debug: print dataset stats
    print("=== Dataset Y statistics ===")
    print("Mean:", np.array(Y_mean))
    print("Std: ", np.array(Y_std))
    print("Var per component:", np.var(np.array(Y_train), axis=0))
    print("============================")

    # --- Init model ---
    model_layers = list(cfg.model.layers)
    model_layers[-1] = 6
    key = jax.random.PRNGKey(cfg.seed)
    model = MLP(features=model_layers, dropout=cfg.model.dropout, activation_fn=nn.relu)
    dummy_input = jnp.ones([1, X_train.shape[1]])
    params = model.init(key, dummy_input)

    optimizer = optax.adam(cfg.training.learning_rate)
    opt_state = optimizer.init(params)

    lambda_phys = cfg.training.lambda_phys
    train_step = make_train_step(model, optimizer, lambda_phys, X_mean, X_std, Y_mean, Y_std)

    # --- Logging ---
    log_dir = os.path.join("jax_logs", "pinn_maxwellB")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, "version_0"))

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    batch_size = cfg.training.batch_size
    num_batches = int(np.ceil(X_train.shape[0] / batch_size))

    for epoch in range(cfg.training.num_epochs):
        perm = np.random.permutation(X_train.shape[0])
        X_shuf, Y_shuf = X_train[perm], Y_train[perm]
        total_loss_ep, total_dloss, total_ploss = 0, 0, 0
        dropout_key = jax.random.fold_in(key, epoch)

        for i in range(num_batches):
            s, e = i * batch_size, min((i+1) * batch_size, X_train.shape[0])
            xb, yb = X_shuf[s:e], Y_shuf[s:e]
            params, opt_state, loss_val, d_loss, p_loss = train_step(params, opt_state, xb, yb, dropout_key)
            total_loss_ep += loss_val.item() * (e - s)
            total_dloss += d_loss.item() * (e - s)
            total_ploss += p_loss.item() * (e - s)

        avg_total_loss = total_loss_ep / X_train.shape[0]
        avg_data_loss = total_dloss / X_train.shape[0]
        avg_phys_loss = total_ploss / X_train.shape[0]

        train_losses.append(avg_total_loss)

        # Validation
        avg_val_loss, (val_data_loss, val_phys_loss) = compute_losses(
            params, model, X_val, Y_val, lambda_phys, train=False,
            X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std
        )
        avg_val_loss, val_data_loss, val_phys_loss = map(float, [avg_val_loss, val_data_loss, val_phys_loss])
        val_losses.append(avg_val_loss)

        # Predictions in physical space
        y_val_pred_norm = np.array(model.apply(params, X_val, train=False))
        y_val_true_norm = np.array(Y_val)
        y_val_pred_phys = y_val_pred_norm * np.array(Y_std) + np.array(Y_mean)
        y_val_true_phys = y_val_true_norm * np.array(Y_std) + np.array(Y_mean)

        val_mae = mean_absolute_error(y_val_true_phys, y_val_pred_phys)
        val_r2 = r2_score(y_val_true_phys, y_val_pred_phys)

        # Per-component metrics
        comp_r2 = [r2_score(y_val_true_phys[:, i], y_val_pred_phys[:, i]) for i in range(6)]
        comp_mae = [mean_absolute_error(y_val_true_phys[:, i], y_val_pred_phys[:, i]) for i in range(6)]
        print(f"Val R² per comp: {', '.join(f'{r:.4f}' for r in comp_r2)}")
        print(f"Val MAE per comp: {', '.join(f'{m:.3e}' for m in comp_mae)}")

        # Log to TensorBoard
        writer.add_scalar("Train/total_loss", avg_total_loss, epoch)
        writer.add_scalar("Train/data_loss", avg_data_loss, epoch)
        writer.add_scalar("Train/physics_loss", avg_phys_loss, epoch)
        writer.add_scalar("Val/total_loss", avg_val_loss, epoch)
        writer.add_scalar("Val/data_loss", val_data_loss, epoch)
        writer.add_scalar("Val/physics_loss", val_phys_loss, epoch)
        writer.add_scalar("Val/MAE", val_mae, epoch)
        writer.add_scalar("Val/R2", val_r2, epoch)

        print(f"Epoch {epoch+1} | Train Loss={avg_total_loss:.3e} | Val Loss={avg_val_loss:.3e} | Val R²={val_r2:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_checkpoint(params, X_mean, X_std, Y_mean, Y_std, os.path.join(cfg.output_dir, "pinn_maxwellB", "trained_params.msgpack"))
        else:
            patience_counter += 1
            if patience_counter >= cfg.training.patience:
                break

    writer.close()

    # Plot curves
    fig_dir = os.path.join(cfg.output_dir, "pinn_maxwellB", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_learning_curves(train_losses, val_losses, fig_dir, cfg.model_type)

    # --- Test Eval ---
    restored = load_checkpoint(os.path.join(cfg.output_dir, "pinn_maxwellB", "trained_params.msgpack"),
        {"params": params, "X_mean": X_mean, "X_std": X_std, "Y_mean": Y_mean, "Y_std": Y_std})
    best_params = restored["params"]

    y_pred_norm = np.array(model.apply(best_params, X_test, train=False))
    y_pred_phys = y_pred_norm * np.array(Y_std) + np.array(Y_mean)
    y_true_phys = np.array(Y_test) * np.array(Y_std) + np.array(Y_mean)

    residuals = y_true_phys - y_pred_phys
    plot_residual_hist(residuals, fig_dir, cfg.model_type)
    plot_residuals_vs_pred(y_pred_phys, residuals, fig_dir, cfg.model_type)

    test_total_loss_norm, (test_data_loss_norm, test_phys_loss_norm) = compute_losses(
        best_params, model, X_test, Y_test, lambda_phys, train=False,
        X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std
    )
    test_mse_phys = np.mean((y_pred_phys - y_true_phys) ** 2)
    test_r2_phys  = r2_score(y_true_phys, y_pred_phys)
    test_comp_r2 = [r2_score(y_true_phys[:, i], y_pred_phys[:, i]) for i in range(6)]
    test_comp_mae = [mean_absolute_error(y_true_phys[:, i], y_pred_phys[:, i]) for i in range(6)]
    print(f"Test R² per comp: {', '.join(f'{r:.4f}' for r in test_comp_r2)}")
    print(f"Test MAE per comp: {', '.join(f'{m:.3e}' for m in test_comp_mae)}")

    metrics_table = [
        ["Train/total_loss", avg_total_loss],
        ["Train/data_loss", avg_data_loss],
        ["Train/physics_loss", avg_phys_loss],
        ["Val/total_loss", avg_val_loss],
        ["Val/data_loss", val_data_loss],
        ["Val/physics_loss", val_phys_loss],
        ["Val/MAE", val_mae],
        ["Val/R2", val_r2],
        ["Test/total_loss", test_total_loss_norm],
        ["Test/data_loss", test_data_loss_norm],
        ["Test/physics_loss", test_phys_loss_norm],
        ["Test/MSE", test_mse_phys],
        ["Test/R2", test_r2_phys]
    ]
    print("\n" + tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="grid"))

if __name__ == "__main__":
    main()