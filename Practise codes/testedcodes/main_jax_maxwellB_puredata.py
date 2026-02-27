import os
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from omegaconf import DictConfig
import hydra
from sklearn.metrics import r2_score, mean_absolute_error
from tabulate import tabulate

# === Custom data loader for Maxwell-B
from utils.load_and_normalize_data_maxwellB import load_and_normalize_data_maxwellB as load_and_normalize_data
from utils.train_utils import save_checkpoint, load_checkpoint, plot_learning_curves

# Maxwell-B constants
ETA0 = 5.28e-5
LAM = 1.902

# === Invariants computation ===
def compute_invariants(L):
    """Compute basic isotropic invariants from velocity gradient L (3x3)."""
    D = 0.5 * (L + L.T)
    W = 0.5 * (L - L.T)
    I1 = np.trace(D)
    I2 = 0.5 * (np.trace(D)**2 - np.trace(D @ D))
    I3 = np.linalg.det(D)
    J2 = -0.5 * np.trace(W @ W)           # second invariant of W
    K1 = np.trace(D @ (W @ W))            # coupling invariant
    return np.array([I1, I2, I3, J2, K1])

# --- Tensor conversion helper ---
def vec6_to_sym3(vec):
    T = jnp.zeros((vec.shape[0], 3, 3))
    T = T.at[:, 0, 0].set(vec[:, 0])
    T = T.at[:, 1, 1].set(vec[:, 1])
    T = T.at[:, 2, 2].set(vec[:, 2])
    T = T.at[:, 0, 1].set(vec[:, 3]); T = T.at[:, 1, 0].set(vec[:, 3])
    T = T.at[:, 0, 2].set(vec[:, 4]); T = T.at[:, 2, 0].set(vec[:, 4])
    T = T.at[:, 1, 2].set(vec[:, 5]); T = T.at[:, 2, 1].set(vec[:, 5])
    return T

# --- Maxwell-B residual ---
def maxwellB_residual(L_flat, T_pred_flat):
    L = L_flat.reshape(-1, 3, 3)
    T = vec6_to_sym3(T_pred_flat)
    D = 0.5 * (L + jnp.swapaxes(L, 1, 2))
    LTt = jnp.matmul(jnp.swapaxes(L, 1, 2), T)
    TL  = jnp.matmul(T, L)
    return T - LAM * (LTt + TL) - 2.0 * ETA0 * D

# --- MLP ---
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
        return nn.Dense(self.features[-1])(x)

# --- Loss ---
def compute_losses(params, model, x, y, train=True, dropout_key=None):
    if train and dropout_key is not None:
        preds = model.apply(params, x, train=True, rngs={'dropout': dropout_key})
    else:
        preds = model.apply(params, x, train=False)
    return jnp.mean((preds - y) ** 2)

def make_train_step(model, optimizer):
    @jax.jit
    def train_step(params, opt_state, x, y, dropout_key):
        loss_val, grads = jax.value_and_grad(compute_losses)(params, model, x, y, True, dropout_key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val
    return train_step

@hydra.main(config_path="config/train", config_name="maxwellB_pure_config", version_base=None)
def main(cfg: DictConfig):
    jax.config.update("jax_enable_x64", True)

    # --- Load dataset ---
    X_path, Y_path = "./datafiles/X_3D_maxwell_B.pt", "./datafiles/Y_3D_maxwell_B.pt"
    X_train_raw, X_val_raw, X_test_raw, Y_train, Y_val, Y_test, X_mean, X_std, Y_mean, Y_std = \
        load_and_normalize_data(X_path, Y_path, seed=cfg.seed)

    # --- Denormalize to get physical L (9 comps) for invariants and residual check ---
    L_train_phys = X_train_raw * X_std + X_mean
    L_val_phys   = X_val_raw   * X_std + X_mean
    L_test_phys  = X_test_raw  * X_std + X_mean

    # --- Convert physical L to invariants ---
    X_train_inv = np.stack([compute_invariants(L.reshape(3,3)) for L in L_train_phys])
    X_val_inv   = np.stack([compute_invariants(L.reshape(3,3)) for L in L_val_phys])
    X_test_inv  = np.stack([compute_invariants(L.reshape(3,3)) for L in L_test_phys])

    # --- Normalize invariants ---
    X_mean_inv = X_train_inv.mean(axis=0)
    X_std_inv  = X_train_inv.std(axis=0); X_std_inv[X_std_inv == 0] = 1
    X_train    = (X_train_inv - X_mean_inv) / X_std_inv
    X_val      = (X_val_inv   - X_mean_inv) / X_std_inv
    X_test     = (X_test_inv  - X_mean_inv) / X_std_inv

    # --- Residual check using original L ---
    Y_train_phys = Y_train * Y_std + Y_mean
    resids = maxwellB_residual(L_train_phys[:5], Y_train_phys[:5])
    print("\n=== Residual Norm Check (first 5 samples) ===")
    for i in range(resids.shape[0]):
        print(f"Sample {i}: Fro norm residual = {np.linalg.norm(np.array(resids[i])):.3e}")

    # --- Model & optimizer ---
    model = MLP(features=list(cfg.model.layers))
    key = jax.random.PRNGKey(cfg.seed)
    params = model.init(key, jnp.ones([1, X_train.shape[1]]))
    optimizer = optax.adam(cfg.training.learning_rate)
    opt_state = optimizer.init(params)
    train_step = make_train_step(model, optimizer)

    # --- Training loop ---
    num_batches = int(np.ceil(X_train.shape[0] / cfg.training.batch_size))
    train_losses, val_losses = [], []
    for epoch in range(cfg.training.num_epochs):
        perm = np.random.permutation(X_train.shape[0])
        X_sh, Y_sh = X_train[perm], Y_train[perm]
        total_loss_ep, dropout_key = 0.0, jax.random.fold_in(key, epoch)
        for i in range(num_batches):
            s, e = i * cfg.training.batch_size, min((i+1) * cfg.training.batch_size, X_train.shape[0])
            params, opt_state, loss_val = train_step(params, opt_state, X_sh[s:e], Y_sh[s:e], dropout_key)
            total_loss_ep += loss_val.item() * (e - s)
        avg_train_loss = total_loss_ep / X_train.shape[0]
        train_losses.append(avg_train_loss)
        val_loss = float(compute_losses(params, model, X_val, Y_val, train=False))
        val_losses.append(val_loss)
        y_val_pred_phys = (np.array(model.apply(params, X_val, train=False)) * np.array(Y_std)) + np.array(Y_mean)
        y_val_true_phys = (np.array(Y_val) * np.array(Y_std)) + np.array(Y_mean)
        val_r2 = r2_score(y_val_true_phys, y_val_pred_phys)
        val_mae = mean_absolute_error(y_val_true_phys, y_val_pred_phys)
        print(f"Epoch {epoch+1} | Train Loss={avg_train_loss:.3e} | Val Loss={val_loss:.3e} "
              f"| Val R² Global={val_r2:.4f} | Val MAE Global={val_mae:.3e}")
        save_checkpoint(params, X_mean_inv, X_std_inv, Y_mean, Y_std,
                        os.path.join(cfg.output_dir, "puredata_maxwellB_inv", "best.msgpack"))

    # --- Plot learning curves ---
    fig_dir = os.path.join(cfg.output_dir, "puredata_maxwellB_inv", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_learning_curves(train_losses, val_losses, fig_dir, cfg.model_type)

    # --- Final test evaluation ---
    restored = load_checkpoint(os.path.join(cfg.output_dir, "puredata_maxwellB_inv", "best.msgpack"),
                               {"params": params, "X_mean": X_mean_inv, "X_std": X_std_inv,
                                "Y_mean": Y_mean, "Y_std": Y_std})
    best_params = restored["params"]
    y_test_pred_phys = (np.array(model.apply(best_params, X_test, train=False)) * np.array(Y_std)) + np.array(Y_mean)
    y_test_true_phys = (np.array(Y_test) * np.array(Y_std)) + np.array(Y_mean)
    test_r2 = r2_score(y_test_true_phys, y_test_pred_phys)
    test_mae = mean_absolute_error(y_test_true_phys, y_test_pred_phys)
    test_mse = np.mean((y_test_true_phys - y_test_pred_phys) ** 2)
    metrics_table = [
        ["Train/total_loss", float(train_losses[-1])],
        ["Val/total_loss", float(val_losses[-1])],
        ["Val/MAE", float(val_mae)],
        ["Val/R2", float(val_r2)],
        ["Test/MSE", float(test_mse)],
        ["Test/MAE", float(test_mae)],
        ["Test/R2", float(test_r2)],
    ]
    print("\n" + tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="grid"))

if __name__ == "__main__":
    main()