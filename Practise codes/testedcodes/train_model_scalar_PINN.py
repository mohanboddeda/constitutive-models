import os
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import hydra
import numpy as np
import random
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
# Physics loss function (JAX)
# ============================
def carreau_yasuda_viscosity_jax(L,
                                 nu_0=5.28e-5, nu_inf=3.30e-6,
                                 lambda_val=1.902, n=0.22, a=1.25):
    """Compute viscosity from Carreau–Yasuda law."""
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


# ============================
# MLP model definition
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
# Training step with PINN loss
# ============================
def make_train_step(model, optimizer, lambda_phys, X_mean, X_std, Y_mean, Y_std):
    @jax.jit
    def total_loss(params, x, y):
        pred = model.apply(params, x)
         # Data loss (normalized)
        data_loss = jnp.mean((pred - y) ** 2)
        # Denormalize input features from normalized space to physical space
        x_phys = x * X_std + X_mean
        # Compute viscosity in physical space using physics law
        nu_phys_physical = carreau_yasuda_viscosity_jax(x_phys)
        # Normalize physics output to the same scale as training targets
        nu_phys_norm = (nu_phys_physical - Y_mean) / Y_std
        # Compute physics loss in normalized space (same units as data_loss)
        physics_loss = jnp.mean((pred.squeeze() - nu_phys_norm.squeeze()) ** 2)
        total = data_loss + lambda_phys * physics_loss
        return total, (data_loss, physics_loss)

    @jax.jit
    def train_step(params, opt_state, x, y):
        (loss_val, (d_loss, p_loss)), grads = jax.value_and_grad(total_loss, has_aux=True)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val, d_loss, p_loss

    def predict_fn(params, x):
        return model.apply(params, x)

    return train_step, total_loss, predict_fn


# ============================
# Main function
# ============================
@hydra.main(config_path="config/train", config_name="carreau_yasuda_pinn", version_base=None)
def main(cfg: DictConfig):
    # Enable double precision
    jax.config.update("jax_enable_x64", True)

    # Load data
    X_path = f"./datafiles/X_3D_{cfg.model_type}.pt"
    Y_path = f"./datafiles/Y_3D_{cfg.model_type}.pt"
    X_train, X_val, X_test, Y_train, Y_val, Y_test, X_mean, X_std, Y_mean, Y_std = load_and_normalize_data(
        X_path, Y_path, seed=cfg.seed
    )

    # Convert to jax arrays
    X_train, X_val, X_test = jnp.array(X_train), jnp.array(X_val), jnp.array(X_test)
    Y_train, Y_val, Y_test = jnp.array(Y_train), jnp.array(Y_val), jnp.array(Y_test)

    # Init model
    model_layers = list(cfg.model.layers)
    key = jax.random.PRNGKey(cfg.seed)
    model = MLP(features=model_layers)
    params = model.init(key, jnp.ones([1, X_train.shape[1]]))

    optimizer = optax.adam(cfg.training.learning_rate)
    opt_state = optimizer.init(params)

    train_step, total_loss, predict_fn = make_train_step(model, optimizer, cfg.training.lambda_phys,
                                                         X_mean, X_std, Y_mean, Y_std)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    fig_dir = os.path.join(cfg.output_dir, "pinn", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.output_dir,"pinn", "trained_params.msgpack")

    batch_size = cfg.training.batch_size
    num_batches = int(np.ceil(X_train.shape[0] / batch_size))

    for epoch in range(cfg.training.num_epochs):
        perm = np.random.permutation(X_train.shape[0])
        X_shuffled, Y_shuffled = X_train[perm], Y_train[perm]

        total_epoch_loss = 0
        total_data_loss = 0
        total_phys_loss = 0

        for i in range(num_batches):
            start, end = i * batch_size, min((i+1) * batch_size, X_train.shape[0])
            xb, yb = X_shuffled[start:end], Y_shuffled[start:end]
            params, opt_state, loss_val, d_loss, p_loss = train_step(params, opt_state, xb, yb)
            total_epoch_loss += loss_val.item() * (end - start)
            total_data_loss += d_loss.item() * (end - start)
            total_phys_loss += p_loss.item() * (end - start)

        avg_total_loss = total_epoch_loss / X_train.shape[0]
        avg_data_loss = total_data_loss / X_train.shape[0]
        avg_phys_loss = total_phys_loss / X_train.shape[0]

        avg_val_loss, (val_data_loss, val_phys_loss) = total_loss(params, X_val, Y_val)
        avg_val_loss, val_data_loss, val_phys_loss = float(avg_val_loss), float(val_data_loss), float(val_phys_loss)

        train_losses.append(avg_total_loss)
        val_losses.append(avg_val_loss)

        # Print every N epochs
        print_every = 10
        if (epoch + 1) % print_every == 0 or epoch == 0:

           print(f"Epoch {epoch+1} | Train: total={avg_total_loss:.3e}, data={avg_data_loss:.3e}, phys={avg_phys_loss:.3e} | Val: total={avg_val_loss:.3e}, data={val_data_loss:.3e}, phys={val_phys_loss:.3e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_checkpoint(params, X_mean, X_std, Y_mean, Y_std, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.training.patience:
                print("Early stopping triggered")
                break

    plot_learning_curves(train_losses, val_losses, fig_dir, f"{cfg.model_type}_PINN")
    # Save last epoch's train and val losses — will use for final print after training
    last_train_total_loss = avg_total_loss
    last_train_data_loss = avg_data_loss
    last_train_phys_loss = avg_phys_loss

    last_val_total_loss = avg_val_loss
    last_val_data_loss = val_data_loss
    last_val_phys_loss = val_phys_loss
    
    # Evaluate best model
    restored = load_checkpoint(ckpt_path, {"params": params, "X_mean": X_mean, "X_std": X_std, "Y_mean": Y_mean, "Y_std": Y_std})
    best_params = restored["params"]

    # --- Compute Test losses in normalized space
    test_total_loss, (test_data_loss, test_phys_loss) = total_loss(best_params, X_test, Y_test)
    test_total_loss = float(test_total_loss)
    test_data_loss = float(test_data_loss)
    test_phys_loss = float(test_phys_loss)

     # --- In physical units
    preds_norm = predict_fn(best_params, X_test)

    # Denormalize back to physical units
    y_pred = np.array(preds_norm) * Y_std + Y_mean
    y_true = np.array(Y_test) * Y_std + Y_mean
    residuals = y_true - y_pred

    plot_residual_hist(residuals, fig_dir, f"{cfg.model_type}_PINN")
    plot_residuals_vs_pred(y_pred, residuals, fig_dir, f"{cfg.model_type}_PINN")

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse_phys = np.mean((y_pred - y_true) ** 2)
    
    print(f"R²: {r2:.4f}, MAE: {mae:.6e}, MSE: {mse_phys:.6e}")

if __name__ == "__main__":
    main()