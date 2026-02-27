#============================================================
# Imports needed to run the script
#============================================================
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

#=================================================================
# 1.Importing Normalized Data and Plotting functions
#=================================================================
from utils.data_utils_stable import (
    load_and_normalize_data_stable,
    save_checkpoint,
    load_checkpoint,plot_all_losses,
    plot_residual_hist,
    plot_residuals_vs_pred,
    plot_stress_tensor_comparison
)

# ==============================================================
# 2. Helper function: convert 6-comp symmetric tensor vector to 3x3
# ==============================================================
def vec6_to_sym3(vec):
    T = jnp.zeros((vec.shape[0], 3, 3))
    T = T.at[:, 0, 0].set(vec[:, 0])
    T = T.at[:, 1, 1].set(vec[:, 1])
    T = T.at[:, 2, 2].set(vec[:, 2])
    T = T.at[:, 0, 1].set(vec[:, 3]); T = T.at[:, 1, 0].set(vec[:, 3])
    T = T.at[:, 0, 2].set(vec[:, 4]); T = T.at[:, 2, 0].set(vec[:, 4])
    T = T.at[:, 1, 2].set(vec[:, 5]); T = T.at[:, 2, 1].set(vec[:, 5])
    return T

#  ==============================================================
# 3. Physics based residuals for different models
#  ==============================================================
def maxwellB_residual(L_phys, T_phys, eta0, lam):
    D = 0.5 * (L_phys + jnp.swapaxes(L_phys, 1, 2))
    dim = L_phys.shape[1]
    I = jnp.eye(dim)
    A = I - lam * L_phys
    B = -lam * jnp.swapaxes(L_phys, 1, 2)
    C = 2.0 * eta0 * D
    R = jnp.matmul(A, T_phys) + jnp.matmul(T_phys, B) - C
    return R

def oldroydB_residual(L_phys, T_phys, eta0, lam, lam_r):
    D = 0.5 * (L_phys + jnp.swapaxes(L_phys, 1, 2))
    dim = L_phys.shape[1]
    I = jnp.eye(dim)
    A = I - lam * L_phys
    B = -lam * jnp.swapaxes(L_phys, 1, 2)
    C = 2 * eta0 * (D - lam_r * jnp.matmul(L_phys, D) - lam_r * jnp.matmul(D, jnp.swapaxes(L_phys, 1, 2)))
    R = jnp.matmul(A, T_phys) + jnp.matmul(T_phys, B) - C
    return R

def ptt_exponential_residual(L_phys, T_phys, eta0, lam, alpha=1.0):
    """
    Steady-state contravariant exponential PTT residual:
    [(1 + psi) I - lam L] T + T (-lam L^T) - 2 eta0 D = 0
    """
    D = 0.5 * (L_phys + jnp.swapaxes(L_phys, 1, 2))
    dim = L_phys.shape[1]
    I = jnp.eye(dim)
    psi_val = jnp.exp(alpha * jnp.trace(T_phys, axis1=1, axis2=2)) - 1.0
    # Expand psi_val to broadcast with I
    A_eff = (1.0 + psi_val)[:, None, None] * I - lam * L_phys
    B_eff = -lam * jnp.swapaxes(L_phys, 1, 2)
    C = 2.0 * eta0 * D
    R = jnp.matmul(A_eff, T_phys) + jnp.matmul(T_phys, B_eff) - C
    return R

#  ==============================================================
# Activation mapping
#  ==============================================================
activation_map = {
    "relu": nn.relu,
    "tanh": nn.tanh,
    "sigmoid": nn.sigmoid
}

#  ==============================================================
# MLP Model
#  ==============================================================
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


#  ==============================================================
# Compute losses
#  ==============================================================
def compute_losses(params, model, x_norm, y_norm, lambda_phys, train, dropout_key,
                   X_mean, X_std, Y_mean, Y_std, residual_fn, eta0, lam, lam_r=1.0):
    
    use_dropout = (train and (dropout_key is not None))
    preds_norm = model.apply( params, x_norm, train=train, rngs={'dropout': dropout_key} if use_dropout else {})

    # 1. Data Loss (calculated in physical units: MSE purely supervised learning)
    preds_phys = preds_norm * Y_std + Y_mean
    y_phys = y_norm * Y_std + Y_mean
    data_loss = jnp.mean((preds_phys - y_phys) ** 2)

    # 2. Pysics Loss (calculated in physical units: MSE based on residuals of governing equations)
    physics_loss_data = 0.0
    if lambda_phys > 0:
        L_phys = x_norm * X_std + X_mean
        T_phys = vec6_to_sym3(preds_phys)
        if residual_fn == maxwellB_residual:
            residuals_data = residual_fn(L_phys.reshape(-1, 3, 3), T_phys, eta0, lam)
        else:
            residuals_data = residual_fn(L_phys.reshape(-1, 3, 3), T_phys, eta0, lam, lam_r)
        physics_loss_data = jnp.mean(residuals_data ** 2)

    total_loss = data_loss + lambda_phys * physics_loss_data
    return total_loss, (data_loss, physics_loss_data)

#  ==============================================================
#  Make JIT train step
#  ==============================================================
def make_train_step(model, optimizer, lambda_phys, X_mean, X_std, Y_mean, Y_std, residual_fn,
                    eta0, lam, lam_r=1.0):
    @jax.jit
    def train_step(params, opt_state, x, y, dropout_key):
        (loss_val, (d_loss, p_loss)), grads = jax.value_and_grad(compute_losses, has_aux=True)(
            params, model, x, y,
            lambda_phys, True, dropout_key,
            X_mean, X_std, Y_mean, Y_std,
            residual_fn, eta0, lam, lam_r
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val, d_loss, p_loss
    return train_step

#  ==============================================================
#  Cosine LR schedule
#  ==============================================================
def cosine_annealing_lr(init_lr, T_max_epochs, steps_per_epoch):
    T_max_steps = T_max_epochs * steps_per_epoch
    def schedule_fn(step):
        return init_lr * 0.5 * (1 + jnp.cos(jnp.pi * step / T_max_steps))
    return schedule_fn


#  ==============================================================
#  Main training function
#  ==============================================================

def run_training_for_lambda(cfg, lambda_val):
    start_time = time.time()
    out_dir = os.path.join(cfg.output_dir, f"{cfg.model_type}_stable_lambda_{lambda_val}")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Auto-select the dataset paths depending on the chosen model_type
    if cfg.model_type not in cfg.data.paths:
       raise ValueError(f"Unknown model_type {cfg.model_type}, must be 'maxwell_B' or 'oldroyd_B'")

    x_path = cfg.data.paths[cfg.model_type].x
    y_path = cfg.data.paths[cfg.model_type].y

    print(f"📂 Selected dataset for {cfg.model_type} stable:")
    print(f"X file: {x_path}")
    print(f"Y file: {y_path}")
    # Load data
    X_train, X_val, X_test, Y_train, Y_val, Y_test, X_mean, X_std, Y_mean, Y_std = \
        load_and_normalize_data_stable(
            cfg.model_type,
            x_path,
            y_path,
            seed=cfg.seed,
            scaling_mode=cfg.data.scaling_mode
        )

    # Select residual function
    if cfg.model_type == "maxwell_B":
        residual_fn = maxwellB_residual
    elif cfg.model_type == "oldroyd_B":
        residual_fn = oldroydB_residual
    elif cfg.model_type == "ptt_exponential":
        residual_fn = ptt_exponential_residual
    else:
        raise ValueError(f"Unsupported model_type: {cfg.model_type}")

    # Set up model
    activation_fn = activation_map.get(cfg.model.activation, nn.relu)
    model_layers = list(cfg.model.layers)
    model_layers[-1] = 6
    model = MLP(features=model_layers,
                dropout=cfg.model.dropout,
                activation_fn=activation_fn)

    key = jax.random.PRNGKey(cfg.seed)
    # --- Check if doing transfer learning ---
    if cfg.get("transfer_checkpoint", None):
       print(f"🔄 Loading pretrained weights from {cfg.transfer_checkpoint}")
       # dummy initialize for shape
       params = model.init(key, jnp.ones([1, X_train.shape[1]]))
       restored = load_checkpoint(cfg.transfer_checkpoint,
                               {"params": params,
                                "X_mean": X_mean, "X_std": X_std,
                                "Y_mean": Y_mean, "Y_std": Y_std})
       params = restored["params"]
       X_mean = restored["X_mean"]
       X_std  = restored["X_std"]
       Y_mean = restored["Y_mean"]
       Y_std  = restored["Y_std"]
    else:
       print("🚀 Starting from random initialization")
       params = model.init(key, jnp.ones([1, X_train.shape[1]]))
       
    steps_per_epoch = int(np.ceil(X_train.shape[0] / cfg.training.batch_size))
    lr_schedule_fn = cosine_annealing_lr(cfg.training.learning_rate, cfg.training.num_epochs, steps_per_epoch)
    optimizer = optax.adamw(learning_rate=lr_schedule_fn, weight_decay=cfg.training.weight_decay)
    opt_state = optimizer.init(params)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    train_data_losses, val_data_losses = [], []
    train_phys_losses, val_phys_losses = [], []

    # Training loop
    for epoch in range(cfg.training.num_epochs):
        #progress = epoch / cfg.training.num_epochs
        lambda_curr = lambda_val * (epoch / cfg.training.num_epochs)
        #lambda_curr = lambda_val * (progress ** 2)
        train_step = make_train_step(model, optimizer, lambda_curr,
                                     X_mean, X_std, Y_mean, Y_std,
                                     residual_fn,
                                     cfg.eta0, cfg.lam, cfg.lam_r)

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
        train_data_losses.append(avg_data_loss)
        train_phys_losses.append(avg_phys_loss)

        # Validation losses
        avg_val_loss, (val_data_loss, val_phys_loss) = compute_losses(
            params, model, X_val, Y_val,
            lambda_curr, False, None,
            X_mean, X_std, Y_mean, Y_std,
            residual_fn, cfg.eta0, cfg.lam, cfg.lam_r
        )
        avg_val_loss = float(avg_val_loss)

        val_losses.append(avg_val_loss)
        val_data_losses.append(float(val_data_loss))
        val_phys_losses.append(float(val_phys_loss))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(params, X_mean, X_std, Y_mean, Y_std,
                            os.path.join(out_dir, "trained_params.msgpack"))

        if epoch % 50 == 0 or epoch == cfg.training.num_epochs - 1:
            print(f"[{cfg.model_type} Stable] Epoch {epoch}: λ_phys={lambda_curr:.4f}, Val Loss={avg_val_loss:.6e}")

         
    # NEW — plot total, data, and physics loss separately
    plot_all_losses(train_losses, val_losses,
                train_data_losses, val_data_losses,
                train_phys_losses, val_phys_losses,
                Y_std, fig_dir, cfg.model_type)

    restored = load_checkpoint(os.path.join(out_dir, "trained_params.msgpack"),
                               {"params": params, "X_mean": X_mean, "X_std": X_std,
                                "Y_mean": Y_mean, "Y_std": Y_std})
    best_params = restored["params"]

    def de_normalize(Y_norm):
        return np.array(Y_norm) * np.array(Y_std) + np.array(Y_mean)

    y_true_phys = de_normalize(Y_test)
    y_pred_phys = de_normalize(model.apply(best_params, X_test, train=False))

    plot_stress_tensor_comparison(vec6_to_sym3, y_true_phys, y_pred_phys, [0, 5, 10], fig_dir, cfg.model_type)
    residuals = y_true_phys - y_pred_phys
    plot_residual_hist(residuals, fig_dir, cfg.model_type)
    plot_residuals_vs_pred(y_pred_phys, residuals, fig_dir, cfg.model_type)

    test_total_loss, (test_data_loss_norm, test_phys_loss_norm) = compute_losses(
                                                    best_params, model, X_test, Y_test,
                                                    lambda_val, False, None,
                                                    X_mean, X_std, Y_mean, Y_std,
                                                    residual_fn, cfg.eta0, cfg.lam, cfg.lam_r
                            )
    test_total_loss = float(test_total_loss)
    test_data_loss_norm = float(test_data_loss_norm)
    test_phys_loss_norm = float(test_phys_loss_norm)
    # Calculate physical MSE
    test_mse_phys = float(np.mean((y_true_phys - y_pred_phys) ** 2))
    
    # Metrics table for  printing
    metrics_table = [
       ["Train/total_loss", avg_total_loss],
       ["Train/data_loss", avg_data_loss],
       ["Train/physics_loss", avg_phys_loss],
       ["Val/total_loss", avg_val_loss],
       ["Val/data_loss", val_data_loss],
       ["Val/physics_loss", val_phys_loss],
       ["Test/total_loss", test_total_loss],
       ["Test/data_loss", test_data_loss_norm],
       ["Test/physics_loss", test_phys_loss_norm],
       ["Test/MSE", test_mse_phys],
       ["Test/MAE", mean_absolute_error(y_true_phys, y_pred_phys)]
        ]
    print("\n=== Metrics ===")
    print(tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="grid"))

    # Measure elapsed time
    elapsed_time = time.time() - start_time

    # Detect GPU info
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu_name = gpus[0].name
        print(f"⏱ Model training took approximately {elapsed_time:.2f} seconds "
              f"on a single {gpu_name} GPU")
    else:
        print(f"⏱ Model training took approximately {elapsed_time:.2f} seconds on CPU")
        
    return metrics_table


#  ==============================================================
#  Hydra entry point
#  ==============================================================
@hydra.main(config_path="config/train", config_name="stable_tensor_config", version_base=None)
def main(cfg: DictConfig):
    for target_lambda_phys in cfg.training.lambda_phys:
        run_training_for_lambda(cfg, target_lambda_phys)

if __name__ == "__main__":
    main()