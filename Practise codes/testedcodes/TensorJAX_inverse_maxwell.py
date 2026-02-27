import os
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import time
import GPUtil
from omegaconf import DictConfig
import hydra
from tabulate import tabulate
from sklearn.metrics import mean_absolute_error

from utils.data_utils_stable import (
    load_and_normalize_data_stable,
    save_checkpoint,
    load_checkpoint,
    plot_all_losses,
    plot_residual_hist,
    plot_residuals_vs_pred,
    plot_velocitygradient_tensor_comparison,
    vec9_to_square3
)

# ---------------------------------------------------------
# Symmetric vector → 3×3 tensor helper
# ---------------------------------------------------------
def vec6_to_sym3(vec):
    T = jnp.zeros((vec.shape[0], 3, 3))
    T = T.at[:, 0, 0].set(vec[:, 0])
    T = T.at[:, 1, 1].set(vec[:, 1])
    T = T.at[:, 2, 2].set(vec[:, 2])
    T = T.at[:, 0, 1].set(vec[:, 3]); T = T.at[:, 1, 0].set(vec[:, 3])
    T = T.at[:, 0, 2].set(vec[:, 4]); T = T.at[:, 2, 0].set(vec[:, 4])
    T = T.at[:, 1, 2].set(vec[:, 5]); T = T.at[:, 2, 1].set(vec[:, 5])
    return T

# ---------------------------------------------------------
# Maxwell-B physics residual
# ---------------------------------------------------------
def maxwellB_residual(L_phys, T_phys, eta0, lam):
    D = 0.5 * (L_phys + jnp.swapaxes(L_phys, 1, 2))
    dim = L_phys.shape[1]
    I = jnp.eye(dim)
    A = I - lam * L_phys
    B = -lam * jnp.swapaxes(L_phys, 1, 2)
    C = 2.0 * eta0 * D
    R = jnp.matmul(A, T_phys) + jnp.matmul(T_phys, B) - C
    return R

# ---------------------------------------------------------
# Activation mapping
# ---------------------------------------------------------
activation_map = {
    "relu": nn.relu,
    "tanh": nn.tanh,
    "sigmoid": nn.sigmoid
}

# ---------------------------------------------------------
# Generic MLP model
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# Loss computation for inverse model
# ---------------------------------------------------------
def compute_losses_inverse(params, model, y_norm, x_norm,
                           lambda_phys, train, dropout_key,
                           Y_mean, Y_std, X_mean, X_std,
                           residual_fn, eta0, lam):
    # forward pass
    use_dropout = (train and (dropout_key is not None))
    preds_norm = model.apply(
        params, y_norm, train=train,
        rngs={'dropout': dropout_key} if use_dropout else {}
    )

    preds_phys = preds_norm * X_std + X_mean  # predicted L in physical units
    x_true_phys = x_norm * X_std + X_mean     # target L in physical units

    # data loss
    data_loss = jnp.mean((preds_phys - x_true_phys) ** 2)

    # physics loss
    physics_loss_data = 0.0
    if lambda_phys > 0:
        L_phys = preds_phys.reshape(-1, 3, 3)
        D_pred = 0.5 * (L_phys + jnp.swapaxes(L_phys, 1, 2))
        T_pred_phys = 2.0 * eta0 * D_pred
        residuals_data = residual_fn(L_phys, T_pred_phys, eta0, lam)
        physics_loss_data = jnp.mean(residuals_data ** 2)

    total_loss = data_loss + lambda_phys * physics_loss_data
    return total_loss, (data_loss, physics_loss_data)

# ---------------------------------------------------------
# JIT train step for inverse
# ---------------------------------------------------------
def make_train_step_inverse(model, optimizer, lambda_phys,
                            Y_mean, Y_std, X_mean, X_std,
                            residual_fn, eta0, lam):
    @jax.jit
    def train_step(params, opt_state, y, x, dropout_key):
        (loss_val, (d_loss, p_loss)), grads = jax.value_and_grad(
            compute_losses_inverse, has_aux=True)(
            params, model, y, x,
            lambda_phys, True, dropout_key,
            Y_mean, Y_std, X_mean, X_std,
            residual_fn, eta0, lam
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val, d_loss, p_loss
    return train_step

# ---------------------------------------------------------
# Cosine LR schedule
# ---------------------------------------------------------
def cosine_annealing_lr(init_lr, T_max_epochs, steps_per_epoch):
    T_max_steps = T_max_epochs * steps_per_epoch
    def schedule_fn(step):
        return init_lr * 0.5 * (1 + jnp.cos(jnp.pi * step / T_max_steps))
    return schedule_fn

# ---------------------------------------------------------
# Main Mode 2 inverse training
# ---------------------------------------------------------
def run_inverse_training_maxwell(cfg, lambda_val):
    start_time = time.time()
    out_dir = os.path.join(cfg.output_dir, f"maxwell_inverse_lambda_{lambda_val}")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # load dataset
    x_path = cfg.data.paths["maxwell_B"].x
    y_path = cfg.data.paths["maxwell_B"].y
    X_train, X_val, X_test, Y_train, Y_val, Y_test, X_mean, X_std, Y_mean, Y_std = \
        load_and_normalize_data_stable("maxwell_B", x_path, y_path,
                                       seed=cfg.seed, scaling_mode=cfg.data.scaling_mode)

    # -------------------------------------------------
    # build forward model (from forward config layers)
    fwd_activation_fn = activation_map.get(cfg.model.activation, nn.relu)
    fwd_layers = list(cfg.model.forward_layers)
    forward_model = MLP(features=fwd_layers,
                        dropout=cfg.model.dropout,
                        activation_fn=fwd_activation_fn)

    # load forward checkpoint
    forward_ckpt_path = cfg.model.forward_model_ckpt_path
    forward_restored = load_checkpoint(forward_ckpt_path,
                                       {"params": None,
                                        "X_mean": X_mean, "X_std": X_std,
                                        "Y_mean": Y_mean, "Y_std": Y_std})
    forward_params = forward_restored["params"]

    # generate predicted stresses T_pred from forward model
    print("🔄 Generating T_pred from forward model...")
    T_pred_train = forward_model.apply(forward_params, X_train, train=False)
    T_pred_val   = forward_model.apply(forward_params, X_val,   train=False)
    T_pred_test  = forward_model.apply(forward_params, X_test,  train=False)

    # inverse model data: inputs = T_pred, targets = X_true
    inv_train_inputs  = T_pred_train
    inv_val_inputs    = T_pred_val
    inv_test_inputs   = T_pred_test
    inv_train_targets = X_train
    inv_val_targets   = X_val
    inv_test_targets  = X_test

    # -------------------------------------------------
    # build inverse model (from inverse config layers)
    inv_activation_fn = activation_map.get(cfg.model.activation, nn.relu)
    inv_layers = list(cfg.model.inverse_layers)
    inverse_model = MLP(features=inv_layers,
                        dropout=cfg.model.dropout,
                        activation_fn=inv_activation_fn)

    key = jax.random.PRNGKey(cfg.seed)
    params = inverse_model.init(key, jnp.ones([1, inv_train_inputs.shape[1]]))
    steps_per_epoch = int(np.ceil(inv_train_inputs.shape[0] / cfg.training.batch_size))
    lr_schedule_fn = cosine_annealing_lr(cfg.training.learning_rate, cfg.training.num_epochs, steps_per_epoch)
    optimizer = optax.adamw(learning_rate=lr_schedule_fn, weight_decay=cfg.training.weight_decay)
    opt_state = optimizer.init(params)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_data_losses, val_data_losses = [], []
    train_phys_losses, val_phys_losses = [], []

    residual_fn = maxwellB_residual

    # -------------------------------------------------
    # training loop
    for epoch in range(cfg.training.num_epochs):
        lambda_curr = lambda_val * (epoch / cfg.training.num_epochs)
        train_step = make_train_step_inverse(inverse_model, optimizer, lambda_curr,
                                             Y_mean, Y_std, X_mean, X_std,
                                             residual_fn, cfg.eta0, cfg.lam)
        perm = np.random.permutation(inv_train_inputs.shape[0])
        y_sh, x_sh = inv_train_inputs[perm], inv_train_targets[perm]
        total_loss_ep, total_dloss, total_ploss = 0, 0, 0
        dropout_key = jax.random.fold_in(key, epoch)

        for i in range(steps_per_epoch):
            s, e = i * cfg.training.batch_size, min((i+1) * cfg.training.batch_size, inv_train_inputs.shape[0])
            yb, xb = y_sh[s:e], x_sh[s:e]
            params, opt_state, loss_val, d_loss, p_loss = train_step(params, opt_state, yb, xb, dropout_key)
            total_loss_ep += loss_val.item() * (e - s)
            total_dloss += d_loss.item() * (e - s)
            total_ploss += p_loss.item() * (e - s)

        train_losses.append(total_loss_ep / inv_train_inputs.shape[0])
        train_data_losses.append(total_dloss / inv_train_inputs.shape[0])
        train_phys_losses.append(total_ploss / inv_train_inputs.shape[0])

        avg_val_loss, (val_data_loss, val_phys_loss) = compute_losses_inverse(
            params, inverse_model, inv_val_inputs, inv_val_targets,
            lambda_curr, False, None,
            Y_mean, Y_std, X_mean, X_std,
            residual_fn, cfg.eta0, cfg.lam
        )
        val_losses.append(float(avg_val_loss))
        val_data_losses.append(float(val_data_loss))
        val_phys_losses.append(float(val_phys_loss))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(params, Y_mean, Y_std, X_mean, X_std,
                            os.path.join(out_dir, "inverse_trained_params.msgpack"))

        if epoch % 50 == 0 or epoch == cfg.training.num_epochs - 1:
            print(f"[Maxwell Inverse] Epoch {epoch}: λ_phys={lambda_curr:.4f}, "
                  f"Val Loss={avg_val_loss:.6e}")

    # -------------------------------------------------
    # plot losses
    plot_all_losses(train_losses, val_losses,
                    train_data_losses, val_data_losses,
                    train_phys_losses, val_phys_losses,
                    X_std, fig_dir, "maxwell_inverse")

    # -------------------------------------------------
    # test evaluation
    restored = load_checkpoint(os.path.join(out_dir, "inverse_trained_params.msgpack"),
                               {"params": params, "Y_mean": Y_mean, "Y_std": Y_std,
                                "X_mean": X_mean, "X_std": X_std})
    best_params = restored["params"]

    def de_normalize_X(X_norm):
        return np.array(X_norm) * np.array(X_std) + np.array(X_mean)

    x_true_phys = de_normalize_X(inv_test_targets)
    x_pred_phys = de_normalize_X(inverse_model.apply(best_params, inv_test_inputs, train=False))

    sample_indices = [0, 5, 10]
    plot_velocitygradient_tensor_comparison(vec9_to_square3, x_true_phys, x_pred_phys,
                                            sample_indices, fig_dir, "maxwell_inverse")
    plot_residual_hist(x_true_phys - x_pred_phys, fig_dir, "maxwell_inverse")
    plot_residuals_vs_pred(x_pred_phys, x_true_phys - x_pred_phys, fig_dir, "maxwell_inverse")

    test_total_loss, (test_data_loss_norm, test_phys_loss_norm) = compute_losses_inverse(
        best_params, inverse_model, inv_test_inputs, inv_test_targets,
        lambda_val, False, None,
        Y_mean, Y_std, X_mean, X_std,
        residual_fn, cfg.eta0, cfg.lam
    )
    metrics_table = [
        ["Train/total_loss", train_losses[-1]],
        ["Train/data_loss", train_data_losses[-1]],
        ["Train/physics_loss", train_phys_losses[-1]],
        ["Val/total_loss", val_losses[-1]],
        ["Val/data_loss", val_data_losses[-1]],
        ["Val/physics_loss", val_phys_losses[-1]],
        ["Test/total_loss", float(test_total_loss)],
        ["Test/data_loss", float(test_data_loss_norm)],
        ["Test/physics_loss", float(test_phys_loss_norm)],
        ["Test/MAE", mean_absolute_error(x_true_phys, x_pred_phys)]
    ]
    print("\n=== Inverse Metrics ===")
    print(tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="grid"))

    elapsed_time = time.time() - start_time
    gpus = GPUtil.getGPUs()
    if gpus:
        print(f"⏱ Inverse model training took {elapsed_time:.2f}s on {gpus[0].name}")
    else:
        print(f"⏱ Inverse model training took {elapsed_time:.2f}s on CPU")

    return metrics_table

# hydra entry point
@hydra.main(config_path="config/train", config_name="stable_tensor_inverse_config", version_base=None)
def main(cfg: DictConfig):
    for target_lambda_phys in cfg.training.lambda_phys:
        run_inverse_training_maxwell(cfg, target_lambda_phys)

if __name__ == "__main__":
    main()