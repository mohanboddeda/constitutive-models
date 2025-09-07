import os
import random
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

# ------------------------------------
#   MLP Model
# ------------------------------------
class MLP(nn.Module):
    features: list
    dropout: float = 0.0
    activation_fn: callable = nn.relu

    @nn.compact
    def __call__(self, x, train: bool = True):
        if x.ndim == 3:
            x = x.reshape((x.shape[0], -1))
        for feat in self.features[:-1]:
            x = nn.Dense(feat, kernel_init=nn.initializers.kaiming_uniform())(x)
            x = self.activation_fn(x)
            if self.dropout > 0:
                x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
        return nn.Dense(self.features[-1])(x)

# ------------------------------------
#   Carreau–Yasuda viscosity law
# ------------------------------------
def carreau_yasuda_jax(L,
                       nu_0=5.28e-5, nu_inf=3.30e-6,
                       lambda_val=1.902, n=0.22, a=1.25):
    if L.ndim == 2 and L.shape[1] == 9:
        L = L.reshape((-1, 3, 3))
    D = 0.5 * (L + jnp.swapaxes(L, -1, -2))
    D_sq = jnp.matmul(D, D)
    diag = jnp.diagonal(D_sq, axis1=-2, axis2=-1)
    second_inv_D = jnp.sum(diag, axis=-1)
    epsilon = 1e-12
    shear_rate = jnp.sqrt(2.0 * second_inv_D + epsilon)
    term1 = (lambda_val * shear_rate)**a
    term2 = (1.0 + term1)**((n - 1.0) / a)
    return nu_inf + (nu_0 - nu_inf) * term2

# ------------------------------------
#   Loss computation (always combined)
# ------------------------------------
def compute_losses(params, model, x, y,
                   lambda_phys=1.0,
                   train=True, dropout_key=None,
                   X_mean=None, X_std=None, Y_mean=None, Y_std=None):

    if train and dropout_key is not None:
        preds = model.apply(params, x, train=True, rngs={'dropout': dropout_key})
    else:
        preds = model.apply(params, x, train=False)

    preds_phys = preds * Y_std + Y_mean
    y_phys = y * Y_std + Y_mean
    data_loss = jnp.mean((preds_phys - y_phys) ** 2)

    # Always compute physics
    x_phys = x * X_std + X_mean
    nu_phys = carreau_yasuda_jax(x_phys)
    physics_loss = jnp.mean((preds_phys.squeeze() - nu_phys.squeeze()) ** 2)

    total_loss = data_loss + lambda_phys * physics_loss
    return total_loss, (data_loss, physics_loss)

# ------------------------------------
#   Train step
# ------------------------------------
def make_train_step(model, optimizer, lambda_phys,
                    X_mean, X_std, Y_mean, Y_std):
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

# ------------------------------------
#   CosineAnnealingLR match to PyTorch
# ------------------------------------
def cosine_annealing_lr(init_lr, T_max_epochs, steps_per_epoch):
    T_max_steps = T_max_epochs * steps_per_epoch
    def schedule_fn(step):
        return init_lr * 0.5 * (1 + jnp.cos(jnp.pi * (step / T_max_steps)))
    return schedule_fn

# ------------------------------------
#   MAIN
# ------------------------------------
@hydra.main(config_path="config/train", config_name="carreau_yasuda_config", version_base=None)
def main(cfg: DictConfig):
    # Load data
    X_path = f"./datafiles/X_3D_{cfg.model_type}.pt"
    Y_path = f"./datafiles/Y_3D_{cfg.model_type}.pt"
    X_train, X_val, X_test, Y_train, Y_val, Y_test, X_mean, X_std, Y_mean, Y_std = \
        load_and_normalize_data(X_path, Y_path, seed=cfg.seed)

    X_train, X_val, X_test = jnp.array(X_train), jnp.array(X_val), jnp.array(X_test)
    Y_train, Y_val, Y_test = jnp.array(Y_train), jnp.array(Y_val), jnp.array(Y_test)

    # Init model
    model_layers = list(cfg.model.layers)
    key = jax.random.PRNGKey(cfg.seed)
    model = MLP(features=model_layers, dropout=cfg.model.dropout, activation_fn=nn.relu)
    dummy_input = jnp.ones([1, X_train.shape[1], X_train.shape[2]]) if X_train.ndim == 3 else jnp.ones([1, X_train.shape[1]])
    params = model.init(key, dummy_input)

    # Scheduler to match PyTorch's CosineAnnealingLR(T_max=100)
    steps_per_epoch = int(np.ceil(X_train.shape[0] / cfg.training.batch_size))
    lr_schedule_fn = cosine_annealing_lr(cfg.training.learning_rate, T_max_epochs=100, steps_per_epoch=steps_per_epoch)
    optimizer = optax.adamw(learning_rate=lr_schedule_fn, weight_decay=cfg.training.weight_decay)
    opt_state = optimizer.init(params)

    # Always use physics
    lambda_phys = cfg.training.lambda_phys
    train_step = make_train_step(model, optimizer, lambda_phys, X_mean, X_std, Y_mean, Y_std)

    # Logging dir
    log_base = os.path.join("jax_logs", "pinn_yasuda")
    os.makedirs(log_base, exist_ok=True)
    version_id = f"version_0" if not os.listdir(log_base) else f"version_{len(os.listdir(log_base))}"
    writer = SummaryWriter(log_dir=os.path.join(log_base, version_id))

    ckpt_path = os.path.join(cfg.output_dir, "pinn", "trained_params.msgpack")
    best_val_loss = float('inf')
    patience_counter = 0
    batch_size = cfg.training.batch_size
    num_batches = steps_per_epoch
    last_val_mae = None
    last_val_r2 = None
    train_losses, val_losses = [], [] 

    for epoch in range(cfg.training.num_epochs):
        writer.add_scalar("epoch", epoch, epoch)

        perm = np.random.permutation(X_train.shape[0])
        X_shuffled, Y_shuffled = X_train[perm], Y_train[perm]
        total_epoch_loss, total_data_loss, total_phys_loss = 0, 0, 0
        dropout_key = jax.random.fold_in(key, epoch)

        for i in range(num_batches):
            start, end = i * batch_size, min((i+1) * batch_size, X_train.shape[0])
            xb, yb = X_shuffled[start:end], Y_shuffled[start:end]
            params, opt_state, loss_val, d_loss, p_loss = train_step(params, opt_state, xb, yb, dropout_key)

            total_epoch_loss += loss_val.item() * (end - start)
            total_data_loss += d_loss.item() * (end - start)
            total_phys_loss += p_loss.item() * (end - start)

            # Training step logs
            writer.add_scalar("train_both_losses_step", loss_val.item(), epoch * num_batches + i)
            writer.add_scalar("train_loss_step", d_loss.item(), epoch * num_batches + i)

        # Epoch averages
        avg_total_loss = total_epoch_loss / X_train.shape[0]
        avg_data_loss = total_data_loss / X_train.shape[0]
        avg_phys_loss = total_phys_loss / X_train.shape[0]

        # store for final table
        last_train_total_loss = avg_total_loss
        last_train_data_loss  = avg_data_loss
        last_train_phys_loss  = avg_phys_loss 

        writer.add_scalar("train_both_losses_epoch", avg_total_loss, epoch)
        writer.add_scalar("train_loss_epoch", avg_data_loss, epoch)

        # Validation
        avg_val_loss, (val_data_loss, val_phys_loss) = compute_losses(
            params, model, X_val, Y_val, lambda_phys, train=False,
            X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std
        )
        last_val_total_loss   = float(avg_val_loss)
        last_val_data_loss    = float(val_data_loss)
        last_val_phys_loss    = float(val_phys_loss)

        y_val_pred_norm = np.array(model.apply(params, X_val, train=False))
        y_val_true_norm = np.array(Y_val)

        # Convert back to physical space
        y_val_pred_phys = y_val_pred_norm * np.array(Y_std) + np.array(Y_mean)
        y_val_true_phys = y_val_true_norm * np.array(Y_std) + np.array(Y_mean)

        # Metrics in physical space
        val_mae = mean_absolute_error(y_val_true_phys.reshape(-1), y_val_pred_phys.reshape(-1))
        val_r2 = r2_score(y_val_true_phys.reshape(-1), y_val_pred_phys.reshape(-1))

        # for plotting later
        train_losses.append(avg_total_loss)
        val_losses.append(avg_val_loss)

        # Metrics in NORMALIZED space (MATCH PyTorch)
        #val_mae = mean_absolute_error(y_val_true_norm.reshape(-1), y_val_pred_norm.reshape(-1))
        #val_r2  = r2_score(y_val_true_norm.reshape(-1), y_val_pred_norm.reshape(-1))

        writer.add_scalar("val_loss", float(avg_val_loss), epoch)
        writer.add_scalar("val_mae", float(val_mae), epoch)
        writer.add_scalar("val_physics_loss", float(val_phys_loss), epoch)
        writer.add_scalar("val_r2", float(val_r2), epoch)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{cfg.training.num_epochs} | "
                  f"train_both_losses={avg_total_loss:.3e} | "
                  f"val_loss={avg_val_loss:.3e} | "
                  f"val_r2={val_r2:.4f} | val_mae={val_mae:.6e}")
            
        last_val_mae = val_mae
        last_val_r2 = val_r2

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_checkpoint(params, X_mean, X_std, Y_mean, Y_std, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.training.patience:
                print("Early stopping triggered.")
                break

    writer.close()
    
    # === Plot learning curves ===
    fig_dir = os.path.join(cfg.output_dir, "pinn_yasuda", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_learning_curves(train_losses, val_losses, fig_dir, cfg.model_type) 

    # === Load best model ===
    
    restored = load_checkpoint(ckpt_path, {"params": params, "X_mean": X_mean, "X_std": X_std, "Y_mean": Y_mean, "Y_std": Y_std})
    best_params = restored["params"]
    
    # === Test set evaluation ===
    y_pred = np.array(model.apply(best_params, X_test, train=False))
    y_true = np.array(Y_test)

    y_pred_phys = y_pred * np.array(Y_std) + np.array(Y_mean)
    y_true_phys = y_true * np.array(Y_std) + np.array(Y_mean)

    # Compute residuals for plotting
    residuals = y_true_phys - y_pred_phys

    # Save residual plots
    plot_residual_hist(residuals, fig_dir, cfg.model_type)
    plot_residuals_vs_pred(y_pred_phys, residuals, fig_dir, cfg.model_type)

    # Test metrics in PyTorch space
    test_loss = np.mean((y_pred_phys - y_true_phys) ** 2)  # MSE in physical space
    r2_test = r2_score(y_true_phys.reshape(-1), y_pred_phys.reshape(-1))
    mae_test = mean_absolute_error(y_true_phys.reshape(-1), y_pred_phys.reshape(-1))

    # Metrics in normalized space
    #r2_test = r2_score(y_true.reshape(-1), y_pred.reshape(-1))
    #mae_test = mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1))

    metrics_table = [
    ["Train/total_loss",      f"{last_train_total_loss:.6e}"],
    ["Train/data_loss",       f"{last_train_data_loss:.6e}"],
    ["Train/physics_loss",    f"{last_train_phys_loss:.6e}"],
    ["Val/total_loss",        f"{last_val_total_loss:.6e}"],
    ["Val/data_loss",         f"{last_val_data_loss:.6e}"],
    ["Val/physics_loss",      f"{last_val_phys_loss:.6e}"],
    ["Val/MAE",               f"{last_val_mae:.6e}"],
    ["Val/R2",                f"{last_val_r2:.6f}"],
    ["Test/total_loss (MSE)", f"{test_loss:.6e}"],
    ["Test/MAE",              f"{mae_test:.6e}"],
    ["Test/R2",               f"{r2_test:.6f}"]
    ]
    print("\n" + tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="grid"))

if __name__ == "__main__":
    main()