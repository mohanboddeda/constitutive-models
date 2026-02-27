import os
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import hydra
import numpy as np
import random
from functools import partial
from omegaconf import DictConfig
from sklearn.metrics import r2_score, mean_absolute_error

from utils.train_utils import (
    load_and_normalize_data,
    save_checkpoint,
    load_checkpoint,
    plot_learning_curves,
    plot_residual_hist,
    plot_residuals_vs_pred
)

# ============================
# MLP definition
# ============================
class MLP(nn.Module):
    features: list
    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
        x = nn.Dense(self.features[-1])(x)
        return x

# ============================
# Train step factory
# ============================
def make_train_step(model, optimizer):
    @jax.jit
    def mse_loss(params, x, y):
        preds = model.apply(params, x)
        return jnp.mean((preds - y) ** 2)

    @jax.jit
    def train_step(params, opt_state, x, y):
        loss = mse_loss(params, x, y)
        grads = jax.grad(mse_loss)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    def predict_fn(params, x):
        return model.apply(params, x)

    return train_step, mse_loss, predict_fn

# ============================
# Main function
# ============================
@hydra.main(config_path="config/train", config_name="trainConfigscalar", version_base=None)
def main(cfg: DictConfig):
    
    # Enable double precision
    jax.config.update("jax_enable_x64", True)

    # Data paths
    X_path = f"./datafiles/X_3D_{cfg.model_type}.pt"
    Y_path = f"./datafiles/Y_3D_{cfg.model_type}.pt"

    # Load and normalize data
    X_train, X_val, X_test, Y_train, Y_val, Y_test, X_mean, X_std, Y_mean, Y_std = load_and_normalize_data(
        X_path, Y_path, seed=cfg.seed
    )

    X_train, X_val, X_test = jnp.array(X_train), jnp.array(X_val), jnp.array(X_test)
    Y_train, Y_val, Y_test = jnp.array(Y_train), jnp.array(Y_val), jnp.array(Y_test)

    # Model init
    output_dim = 1
    model_layers = list(cfg.model.layers)
    model_layers[-1] = output_dim
    key = jax.random.PRNGKey(cfg.seed)
    model = MLP(features=model_layers)
    params = model.init(key, jnp.ones([1, X_train.shape[1]]))

    # Learning rate schedule for fine-tuning to very low losses
    steps_per_epoch = int(np.ceil(X_train.shape[0] / cfg.training.batch_size))
    schedule = optax.piecewise_constant_schedule(
        init_value=cfg.training.learning_rate,  # e.g., 1e-3 from yaml
        boundaries_and_scales={
            50 * steps_per_epoch: 0.1,    # -> 1e-4 after epoch 50
            100 * steps_per_epoch: 0.1,   # -> 1e-5 after epoch 100
            150 * steps_per_epoch: 0.1    # -> 1e-6 after epoch 150
        }
    )
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(params)

    train_step, mse_loss, predict_fn = make_train_step(model, optimizer)
    batch_size = cfg.training.batch_size
    num_batches = int(np.ceil(X_train.shape[0] / batch_size))
    
    # Training loop
    best_val_loss = float('inf')
    patience = cfg.training.patience
    patience_counter = 0
    train_losses, val_losses = [], []

    fig_dir = os.path.join(f"./trained_models/{cfg.model_type}", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    ckpt_path = os.path.join(f"./trained_models/{cfg.model_type}", "trained_params.msgpack")

    for epoch in range(cfg.training.num_epochs):
        perm = np.random.permutation(X_train.shape[0])
        X_shuffled, Y_shuffled = X_train[perm], Y_train[perm]

        epoch_loss = 0.0
        for i in range(num_batches):
            start, end = i * batch_size, min((i+1) * batch_size, X_train.shape[0])
            x_batch, y_batch = X_shuffled[start:end], Y_shuffled[start:end]
            params, opt_state, loss = train_step(params, opt_state, x_batch, y_batch)
            epoch_loss += loss.item() * (end - start)

        avg_train_loss = epoch_loss / X_train.shape[0]
        avg_val_loss = float(mse_loss(params, X_val, Y_val))

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_checkpoint(params, X_mean, X_std, Y_mean, Y_std, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    plot_learning_curves(train_losses, val_losses, fig_dir, cfg.model_type)

    # Load best params
    init_bundle = {
        "params": params,
        "X_mean": X_mean,
        "X_std": X_std,
        "Y_mean": Y_mean,
        "Y_std": Y_std
    }
    restored = load_checkpoint(ckpt_path, init_bundle)
    best_params = restored["params"]
    X_mean, X_std, Y_mean, Y_std = restored["X_mean"], restored["X_std"], restored["Y_mean"], restored["Y_std"]

    # -------------------------
    # Final normalized losses
    # -------------------------
    final_train_loss = float(mse_loss(best_params, X_train, Y_train))
    final_val_loss   = float(mse_loss(best_params, X_val, Y_val))
    final_test_loss  = float(mse_loss(best_params, X_test, Y_test))

    # Test evaluation in physical units
    preds_norm = predict_fn(best_params, X_test)
    y_pred = np.array(preds_norm) * Y_std + Y_mean
    y_true = np.array(Y_test) * Y_std + Y_mean
    residuals = y_true - y_pred

    plot_residual_hist(residuals, fig_dir, cfg.model_type)
    plot_residuals_vs_pred(y_pred, residuals, fig_dir, cfg.model_type)

    mse_phys = np.mean((y_pred - y_true) ** 2)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    # Final summary prints
    print(f"\nFinal Losses (normalized) | Train: {final_train_loss:.3e} | Val: {final_val_loss:.3e} | Test: {final_test_loss:.3e}")
    print(f"R²: {r2:.4f}, MAE: {mae:.6e}, MSE (physical): {mse_phys:.6e}")

    # Pick 3 random indices from the test set
    random_indices = random.sample(range(len(y_true)), 3)
    print("\n[Sample Predictions vs True Values]")
    print(f"{'Index':>6} {'ν_true (Pa·s)':>18} {'ν_pred (Pa·s)':>18} {'Abs Error':>18} {'% Error':>18}")
    for idx in random_indices:
        true_val = y_true[idx].item()
        pred_val = y_pred[idx].item()
        abs_error = abs(true_val - pred_val)
        pct_error = abs_error / abs(true_val) * 100 if true_val != 0 else 0.0
        print(f"{idx:6d} {true_val:18.6e} {pred_val:18.6e} {abs_error:18.6e} {pct_error:18.3f}")

if __name__ == "__main__":
    main()