import os
import random
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
import torch
from omegaconf import DictConfig
import hydra
from sklearn.metrics import r2_score, mean_absolute_error
from torch.utils.tensorboard import SummaryWriter
from tabulate import tabulate

def solve_sylvester_jax(A, B, C):
    """
    Solve A X + X B = C for batched inputs using Kronecker + linear solve.
    Shapes:
      A [batch, dim, dim]
      B [batch, dim, dim]
      C [batch, dim, dim]
    Returns:
      X [batch, dim, dim]
    """
    dim = A.shape[-1]
    I = jnp.eye(dim)

    def single_solve(Ai, Bi, Ci):
        M = jnp.kron(I, Ai) + jnp.kron(Bi.T, I)  # (dim*dim, dim*dim)
        rhs = Ci.reshape(-1)
        sol = jnp.linalg.solve(M, rhs)           # (dim*dim,)
        return sol.reshape(dim, dim)

    return jax.vmap(single_solve)(A, B, C)

# ==============================================================
# Utility functions
# ==============================================================

def flatten_symmetric_jax(T):
    """
    Flatten symmetric matrices: take upper triangle (including diagonal).
    T: [..., dim, dim]
    Output: [..., dim*(dim+1)//2]
    """
    dim = T.shape[-1]
    idxs = [(i, j) for i in range(dim) for j in range(i, dim)]
    return jnp.stack([T[..., i, j] for (i, j) in idxs], axis=-1)

def load_and_normalize_data(X_path, Y_path, seed=42, split=(0.8, 0.1, 0.1)):
    """
    Loads X and Y from torch .pt files and splits into train/val/test.
    Also returns normalized data and stats.
    """
    X = torch.load(X_path)
    Y = torch.load(Y_path)
    X = np.array(X)
    Y = np.array(Y)

    # Shuffle & split
    np.random.seed(seed)
    perm = np.random.permutation(X.shape[0])
    X = X[perm]
    Y = Y[perm]

    n_train = int(split[0] * X.shape[0])
    n_val = int(split[1] * X.shape[0])
    train_X, val_X, test_X = np.split(X, [n_train, n_train + n_val])
    train_Y, val_Y, test_Y = np.split(Y, [n_train, n_train + n_val])

    # Normalize
    X_mean, X_std = np.mean(train_X, axis=0), np.std(train_X, axis=0)
    Y_mean, Y_std = np.mean(train_Y, axis=0), np.std(train_Y, axis=0)
    X_std[X_std == 0] = 1.0
    Y_std[Y_std == 0] = 1.0

    train_Xn = (train_X - X_mean) / X_std
    val_Xn   = (val_X   - X_mean) / X_std
    test_Xn  = (test_X  - X_mean) / X_std
    train_Yn = (train_Y - Y_mean) / Y_std
    val_Yn   = (val_Y   - Y_mean) / Y_std
    test_Yn  = (test_Y  - Y_mean) / Y_std

    return (train_Xn, val_Xn, test_Xn,
            train_Yn, val_Yn, test_Yn,
            X_mean, X_std, Y_mean, Y_std)

# ==============================================================
# MLP Model
# ==============================================================

class MLP(nn.Module):
    features: list
    dropout: float = 0.0
    activation_fn: callable = nn.relu

    @nn.compact
    def __call__(self, x, train: bool = True):
        if x.ndim == 3:
            x = x.reshape((x.shape[0], -1))
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = self.activation_fn(x)
            if self.dropout > 0:
                x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
        return nn.Dense(self.features[-1])(x)

# ==============================================================
# Maxwell-B physics law in JAX
# ==============================================================

def maxwell_B_jax(L, eta0=5.28e-5, lam=1.902):
    if L.ndim == 2:
        dim = int(np.sqrt(L.shape[1]))
        L = L.reshape((-1, dim, dim))
    else:
        dim = L.shape[-1]

    D = 0.5 * (L + jnp.swapaxes(L, -1, -2))
    A = (1.0 / lam) * jnp.eye(dim) - jnp.swapaxes(L, -1, -2)
    B = -L
    C = (2.0 * eta0 / lam) * D

    T = solve_sylvester_jax(A, B, C)   # batch-safe solve
    return flatten_symmetric_jax(T)

# ==============================================================
# Loss computation
# ==============================================================

def compute_losses_maxwell(params, model, x, y,
                            lambda_phys=1.0,
                            train=True, dropout_key=None,
                            X_mean=None, X_std=None, Y_mean=None, Y_std=None):

    if train and dropout_key is not None:
        preds = model.apply(params, x, train=True, rngs={'dropout': dropout_key})
    else:
        preds = model.apply(params, x, train=False)

    # --- Data loss on normalized values ---
    data_loss = jnp.mean((preds - y) ** 2)

    # --- Physics loss ---
    x_phys = x * X_std + X_mean
    T_phys = maxwell_B_jax(x_phys)  # physical stress in physical space
    T_phys_norm = (T_phys - Y_mean) / Y_std
    physics_loss = jnp.mean((preds - T_phys_norm) ** 2)

    total_loss = data_loss + lambda_phys * physics_loss
    return total_loss, (data_loss, physics_loss)

def make_train_step_maxwell(model, optimizer, lambda_phys,
                            X_mean, X_std, Y_mean, Y_std):
    @jax.jit
    def train_step(params, opt_state, x, y, dropout_key):
        (loss_val, (d_loss, p_loss)), grads = jax.value_and_grad(compute_losses_maxwell, has_aux=True)(
            params, model, x, y,
            lambda_phys,
            True, dropout_key,
            X_mean, X_std, Y_mean, Y_std
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val, d_loss, p_loss
    return train_step

# ==============================================================
# Main training loop
# ==============================================================

@hydra.main(config_path="config/train", config_name="maxwell_config", version_base=None)
def main(cfg: DictConfig):
    jax.config.update("jax_enable_x64", True)

    # --- Load data ---
    X_path = f"./datafiles/X_{cfg.data.dim}D_{cfg.data.constitutive_eq}.pt"
    Y_path = f"./datafiles/Y_{cfg.data.dim}D_{cfg.data.constitutive_eq}.pt"

    X_train, X_val, X_test, Y_train, Y_val, Y_test, X_mean, X_std, Y_mean, Y_std = \
        load_and_normalize_data(X_path, Y_path, seed=cfg.seed)

    X_train, X_val, X_test = jnp.array(X_train), jnp.array(X_val), jnp.array(X_test)
    Y_train, Y_val, Y_test = jnp.array(Y_train), jnp.array(Y_val), jnp.array(Y_test)

    # --- Init model & optimizer ---
    key = jax.random.PRNGKey(cfg.seed)
    model = MLP(features=list(cfg.model.layers),
                dropout=cfg.model.dropout,
                activation_fn=nn.relu)
    dummy_input = jnp.ones([1, X_train.shape[1]])
    params = model.init(key, dummy_input)

    optimizer = optax.adam(cfg.training.learning_rate)
    opt_state = optimizer.init(params)

    lambda_phys = cfg.training.lambda_phys
    train_step = make_train_step_maxwell(model, optimizer, lambda_phys,
                                         X_mean, X_std, Y_mean, Y_std)

    # --- Logging ---
    log_base = os.path.join("jax_logs", "pinn_maxwell")
    os.makedirs(log_base, exist_ok=True)
    version_id = f"version_{len(os.listdir(log_base))}"
    writer = SummaryWriter(log_dir=os.path.join(log_base, version_id))

    # --- Training loop ---
    best_val_loss = float('inf')
    patience_counter = 0
    batch_size = cfg.training.batch_size
    num_batches = int(np.ceil(X_train.shape[0] / batch_size))
    train_losses, val_losses = [], []

    for epoch in range(cfg.training.num_epochs):
        perm = np.random.permutation(X_train.shape[0])
        X_shuffled, Y_shuffled = X_train[perm], Y_train[perm]
        total_epoch_loss, total_data_loss, total_phys_loss = 0, 0, 0
        dropout_key_epoch = jax.random.fold_in(key, epoch)

        for i in range(num_batches):
            start, end = i * batch_size, min((i+1) * batch_size, X_train.shape[0])
            xb, yb = X_shuffled[start:end], Y_shuffled[start:end]
            dropout_key_batch, _ = jax.random.split(dropout_key_epoch)
            params, opt_state, loss_val, d_loss, p_loss = train_step(params, opt_state, xb, yb, dropout_key_batch)
            total_epoch_loss += loss_val.item() * (end - start)
            total_data_loss += d_loss.item() * (end - start)
            total_phys_loss += p_loss.item() * (end - start)

        avg_total_loss = total_epoch_loss / X_train.shape[0]
        avg_data_loss = total_data_loss / X_train.shape[0]
        avg_phys_loss = total_phys_loss / X_train.shape[0]
        train_losses.append(avg_total_loss)

        # --- Validation ---
        avg_val_loss, (val_data_loss, val_phys_loss) = compute_losses_maxwell(
            params, model, X_val, Y_val, lambda_phys, train=False,
            X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std
        )
        val_losses.append(float(avg_val_loss))

        # --- Convert back to physical space for metrics ---
        y_val_pred_norm = np.array(model.apply(params, X_val, train=False))
        y_val_true_norm = np.array(Y_val)
        y_val_pred_phys = y_val_pred_norm * np.array(Y_std) + np.array(Y_mean)
        y_val_true_phys = y_val_true_norm * np.array(Y_std) + np.array(Y_mean)
        last_val_mae = mean_absolute_error(y_val_true_phys, y_val_pred_phys)
        last_val_r2  = r2_score(y_val_true_phys, y_val_pred_phys)

        # --- Logging ---
        writer.add_scalar("Train/total_loss", avg_total_loss, epoch)
        writer.add_scalar("Train/data_loss", avg_data_loss, epoch)
        writer.add_scalar("Train/physics_loss", avg_phys_loss, epoch)
        writer.add_scalar("Val/total_loss", float(avg_val_loss), epoch)
        writer.add_scalar("Val/data_loss", float(val_data_loss), epoch)
        writer.add_scalar("Val/physics_loss", float(val_phys_loss), epoch)
        writer.add_scalar("Val/MAE", last_val_mae, epoch)
        writer.add_scalar("Val/R2", last_val_r2, epoch)

        # --- Early stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.training.patience:
                break
        # === After training, compute final metrics ===
    avg_val_loss, (val_data_loss, val_phys_loss) = compute_losses_maxwell(
        params, model, X_val, Y_val, lambda_phys, train=False,
        X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std
    )

    # Convert predictions to physical space
    y_val_pred_norm = np.array(model.apply(params, X_val, train=False))
    y_val_true_norm = np.array(Y_val)
    y_val_pred_phys = y_val_pred_norm * np.array(Y_std) + np.array(Y_mean)
    y_val_true_phys = y_val_true_norm * np.array(Y_std) + np.array(Y_mean)

    # Compute metrics
    val_mae = mean_absolute_error(y_val_true_phys, y_val_pred_phys)
    val_r2  = r2_score(y_val_true_phys, y_val_pred_phys)

    # Do the same for TEST set
    test_total_loss, (test_data_loss, test_phys_loss) = compute_losses_maxwell(
        params, model, X_test, Y_test, lambda_phys, train=False,
        X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std
    )
    y_test_pred_norm = np.array(model.apply(params, X_test, train=False))
    y_test_true_norm = np.array(Y_test)
    y_test_pred_phys = y_test_pred_norm * np.array(Y_std) + np.array(Y_mean)
    y_test_true_phys = y_test_true_norm * np.array(Y_std) + np.array(Y_mean)
    test_mae = mean_absolute_error(y_test_true_phys, y_test_pred_phys)
    test_r2  = r2_score(y_test_true_phys, y_test_pred_phys)

    # === Print table ===
    from tabulate import tabulate
    metrics_table = [
        ["Val/total_loss",     float(avg_val_loss)],
        ["Val/data_loss",      float(val_data_loss)],
        ["Val/physics_loss",   float(val_phys_loss)],
        ["Val/MAE",            float(val_mae)],
        ["Val/R2",             float(val_r2)],
        ["Test/total_loss",    float(test_total_loss)],
        ["Test/data_loss",     float(test_data_loss)],
        ["Test/physics_loss",  float(test_phys_loss)],
        ["Test/MAE",           float(test_mae)],
        ["Test/R2",            float(test_r2)],
    ]
    print("\n" + tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="grid"))

    writer.close()
    print("\nTraining complete. Best val loss:", best_val_loss)
    writer.close()
    # print final metrics
    print("\nTraining complete. Best val loss:", best_val_loss)

if __name__ == "__main__":
    main()