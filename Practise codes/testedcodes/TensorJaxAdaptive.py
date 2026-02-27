#============================================================
# 0. Imports
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

#============================================================
# 1. Import normalized stage-wise data loader + plotting utils
#============================================================
from utils.data_utils_stable import (
    load_and_normalize_stagewise_data_stable,
    save_checkpoint,
    load_checkpoint,
    plot_all_losses,
    plot_residual_hist,
    plot_residuals_vs_pred,
    plot_stress_tensor_comparison,
    plot_dataset_predictions_summary
)

#============================================================
# 2. Helpers :convert 6-comp symmetric tensor vector to 3x3 matrix
#============================================================
def vec6_to_sym3(vec):
    T = jnp.zeros((vec.shape[0], 3, 3))
    T = T.at[:, 0, 0].set(vec[:, 0])
    T = T.at[:, 1, 1].set(vec[:, 1])
    T = T.at[:, 2, 2].set(vec[:, 2])
    T = T.at[:, 0, 1].set(vec[:, 3]); T = T.at[:, 1, 0].set(vec[:, 3])
    T = T.at[:, 0, 2].set(vec[:, 4]); T = T.at[:, 2, 0].set(vec[:, 4])
    T = T.at[:, 1, 2].set(vec[:, 5]); T = T.at[:, 2, 1].set(vec[:, 5])
    return T

#==============================================================
# 3. Physics based residuals for different models
#==============================================================
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
    D = 0.5 * (L_phys + jnp.swapaxes(L_phys, 1, 2))
    dim = L_phys.shape[1]
    I = jnp.eye(dim)
    psi_val = jnp.exp(alpha * jnp.trace(T_phys, axis1=1, axis2=2)) - 1.0
    A_eff = (1.0 + psi_val)[:, None, None] * I - lam * L_phys
    B_eff = -lam * jnp.swapaxes(L_phys, 1, 2)
    C = 2.0 * eta0 * D
    R = jnp.matmul(A_eff, T_phys) + jnp.matmul(T_phys, B_eff) - C
    return R

#==============================================================
# 4. Activation mapping
#==============================================================
activation_map = {
    "relu": nn.relu,
    "tanh": nn.tanh,
    "sigmoid": nn.sigmoid
}

#==============================================================
# 5. MLP Model
#==============================================================
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

#============================================================
# 6. Adaptive Loss Helper
#============================================================
def compute_adaptive_loss(data_loss, phys_loss, lambda_state, alpha=0.9):
    """
    Automatically adjusts lambda so Physics Loss ~= Data Loss magnitude.
    """
    # Detach gradients so we don't differentiate through the weight update itself
    d_loss_detached = jax.lax.stop_gradient(data_loss)
    p_loss_detached = jax.lax.stop_gradient(phys_loss)
    
    # Calculate the ratio needed to make Terms equal
    # current_ratio * phys_loss = data_loss  => current_ratio = data / phys
    current_ratio = d_loss_detached / (p_loss_detached + 1e-10)
    
    # Update Lambda using Exponential Moving Average (EMA) for stability
    new_lambda = alpha * lambda_state + (1.0 - alpha) * current_ratio
    
    # Use the computed lambda (treated as constant for gradient purposes)
    lambda_fixed = jax.lax.stop_gradient(new_lambda)
    total_loss = data_loss + lambda_fixed * phys_loss
    
    return total_loss, new_lambda

#============================================================
# 7. Compute Losses
#============================================================
def compute_losses(params, model, x_norm, y_norm, lambda_state, train, dropout_key,
                   X_mean, X_std, Y_mean, Y_std, residual_fn, eta0, lam, lam_r=1.0):
    
    use_dropout = (train and (dropout_key is not None))
    preds_norm = model.apply(params, x_norm, train=train,
                             rngs={'dropout': dropout_key} if use_dropout else {})
    
    # 1. Data Loss (Physical Units)
    preds_phys = preds_norm * Y_std + Y_mean
    y_phys = y_norm * Y_std + Y_mean
    data_loss = jnp.mean((preds_phys - y_phys) ** 2)

    # 2. Physics Loss (Physical Units)
    physics_loss_data = 0.0
    L_phys = x_norm * X_std + X_mean
    T_phys = vec6_to_sym3(preds_phys)
    
    if residual_fn == maxwellB_residual:
        residuals_data = residual_fn(L_phys.reshape(-1, 3, 3), T_phys, eta0, lam)
    else:
        residuals_data = residual_fn(L_phys.reshape(-1, 3, 3), T_phys, eta0, lam, lam_r)
        
    physics_loss_data = jnp.mean(residuals_data ** 2)

    # 3. Adaptive Balancing
    if train:
        total_loss, new_lambda = compute_adaptive_loss(data_loss, physics_loss_data, lambda_state)
    else:
        # Validation: use current lambda state, do not update
        total_loss = data_loss + lambda_state * physics_loss_data
        new_lambda = lambda_state

    return total_loss, (data_loss, physics_loss_data, new_lambda)

#==============================================================
# 8. Training step function
#==============================================================
def make_train_step(model, optimizer, X_mean, X_std, Y_mean, Y_std, residual_fn,
                    eta0, lam, lam_r=1.0):
    @jax.jit
    def train_step(params, opt_state, lambda_state, x, y, dropout_key):
        # We pass lambda_state IN, and get updated new_lambda OUT
        (loss_val, (d_loss, p_loss, new_lambda)), grads = jax.value_and_grad(compute_losses, has_aux=True)(
            params, model, x, y,
            lambda_state, True, dropout_key,
            X_mean, X_std, Y_mean, Y_std,
            residual_fn, eta0, lam, lam_r
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, new_lambda, loss_val, d_loss, p_loss
    return train_step

#==============================================================
# 9. Cosine LR schedule
#==============================================================
def cosine_annealing_lr(init_lr, T_max_epochs, steps_per_epoch):
    T_max_steps = T_max_epochs * steps_per_epoch
    def schedule_fn(step):
        return init_lr * 0.5 * (1 + jnp.cos(jnp.pi * step / T_max_steps))
    return schedule_fn

#============================================================
# 10. Main training function
#============================================================
def run_training_for_lambda(cfg, initial_lambda_val,
                            X_train, X_val, X_test,
                            Y_train, Y_val, Y_test,
                            X_mean, X_std, Y_mean, Y_std,
                            stage_tag,
                            transfer_ckpt_path=None):
    start_time = time.time()
    out_dir = os.path.join(cfg.output_dir,
                           f"{cfg.model_type}_stage_{stage_tag}_adaptive")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Select Residual function
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

    # Initialize Lambda State (Adaptive Weight)
    # We start with a reasonable guess (e.g. 1.0 or 1000.0)
    # Ideally, it will auto-correct in the first few batches.
    lambda_state = 1000.0 

    # Transfer learning if checkpoint provided
    if transfer_ckpt_path:
        print(f"🔄 Loading pretrained weights from {transfer_ckpt_path}")
        params = model.init(key, jnp.ones([1, X_train.shape[1]]))
        restored = load_checkpoint(transfer_ckpt_path,
                                   {"params": params,
                                    "X_mean": X_mean, "X_std": X_std,
                                    "Y_mean": Y_mean, "Y_std": Y_std})
        params = restored["params"]
        X_mean = restored["X_mean"]
        X_std  = restored["X_std"]
        Y_mean = restored["Y_mean"]
        Y_std  = restored["Y_std"]
        print(f"🛡️ Transfer Mode: Adaptive balancing active. Starting lambda guess: {lambda_state}")
    else:
        print("🚀 Starting from random initialization")
        params = model.init(key, jnp.ones([1, X_train.shape[1]]))
        print(f"📈 Curriculum Mode: Adaptive balancing active. Starting lambda guess: {lambda_state}")

    steps_per_epoch = int(np.ceil(X_train.shape[0] / cfg.training.batch_size))
    lr_schedule_fn = cosine_annealing_lr(cfg.training.learning_rate,
                                         cfg.training.num_epochs, steps_per_epoch)
    optimizer = optax.adamw(learning_rate=lr_schedule_fn,
                            weight_decay=cfg.training.weight_decay)
    opt_state = optimizer.init(params)

    # Append losses for plotting
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_data_losses, val_data_losses = [], []
    train_phys_losses, val_phys_losses = [], []
    lambda_history = []

    # Compile Train Step (Lambda is now passed as state)
    train_step = make_train_step(model, optimizer, 
                                 X_mean, X_std, Y_mean, Y_std,
                                 residual_fn, cfg.eta0, cfg.lam, cfg.lam_r)

    # Training loop per stage
    for epoch in range(cfg.training.num_epochs):
        
        perm = np.random.permutation(X_train.shape[0])
        X_sh, Y_sh = X_train[perm], Y_train[perm]
        total_loss_ep, total_dloss, total_ploss = 0, 0, 0
        dropout_key = jax.random.fold_in(key, epoch)

        for i in range(steps_per_epoch):
            s, e = i * cfg.training.batch_size, min((i+1) * cfg.training.batch_size, X_train.shape[0])
            xb, yb = X_sh[s:e], Y_sh[s:e]
            
            # --- ADAPTIVE STEP ---
            # We pass lambda_state IN, and receive updated lambda_state OUT
            params, opt_state, lambda_state, loss_val, d_loss, p_loss = train_step(
                params, opt_state, lambda_state, xb, yb, dropout_key
            )
            
            total_loss_ep += loss_val.item() * (e - s)
            total_dloss += d_loss.item() * (e - s)
            total_ploss += p_loss.item() * (e - s)

        avg_total_loss = total_loss_ep / X_train.shape[0]
        avg_data_loss = total_dloss / X_train.shape[0]
        avg_phys_loss = total_ploss / X_train.shape[0]
        
        train_losses.append(avg_total_loss)
        train_data_losses.append(avg_data_loss)
        train_phys_losses.append(avg_phys_loss)
        lambda_history.append(lambda_state)

        # Validation losses
        avg_val_loss, (val_data_loss, val_phys_loss, _) = compute_losses(
            params, model, X_val, Y_val,
            lambda_state, False, None,
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
            print(f"[Stage {stage_tag}] Epoch {epoch}: Val Loss={avg_val_loss:.6e} | λ_adaptive={lambda_state:.1f}")

    # Plotting different plots
    plot_all_losses(train_losses, val_losses,
                    train_data_losses, val_data_losses,
                    train_phys_losses, val_phys_losses,
                    Y_std, fig_dir, cfg.model_type)

    restored = load_checkpoint(os.path.join(out_dir, "trained_params.msgpack"),
                               {"params": params, "X_mean": X_mean, "X_std": X_std,
                                "Y_mean": Y_mean, "Y_std": Y_std})
    best_params = restored["params"]

    y_true_phys = np.array(Y_test) * np.array(Y_std) + np.array(Y_mean)
    y_pred_phys = np.array(model.apply(best_params, X_test, train=False)) * np.array(Y_std) + np.array(Y_mean)

    plot_stress_tensor_comparison(vec6_to_sym3, y_true_phys, y_pred_phys, [0, 5, 10], fig_dir, cfg.model_type)
    plot_dataset_predictions_summary(vec6_to_sym3, y_true_phys, y_pred_phys, fig_dir, cfg.model_type)
    residuals = y_true_phys - y_pred_phys
    plot_residual_hist(residuals, fig_dir, cfg.model_type)
    plot_residuals_vs_pred(y_pred_phys, residuals, fig_dir, cfg.model_type)

    # Calculate Test Losses
    test_total_loss, (test_data_loss_norm, test_phys_loss_norm, _) = compute_losses(
        best_params, model, X_test, Y_test,
        lambda_state, False, None,
        X_mean, X_std, Y_mean, Y_std,
        residual_fn, cfg.eta0, cfg.lam, cfg.lam_r
    )
    
    # Save final lambda used
    final_lambda = float(lambda_state)
    
    metrics_table = [
       ["Train/total_loss", avg_total_loss],
       ["Train/data_loss", avg_data_loss],
       ["Train/physics_loss", avg_phys_loss],
       ["Val/total_loss", avg_val_loss],
       ["Val/data_loss", val_data_loss],
       ["Val/physics_loss", val_phys_loss],
       ["Test/total_loss", float(test_total_loss)],
       ["Test/data_loss", float(test_data_loss_norm)],
       ["Test/physics_loss", float(test_phys_loss_norm)],
       ["Final/Lambda_Adaptive", final_lambda],
       ["Test/MSE", float(np.mean((y_true_phys - y_pred_phys) ** 2))],
       ["Test/MAE", mean_absolute_error(y_true_phys, y_pred_phys)]
        ]
    
    metrics_str = tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="grid")
    metrics_path = os.path.join(out_dir, f"metrics_stage_{stage_tag}.txt")
    with open(metrics_path, "w") as f:
        f.write(f"=== Metrics for Stage: {stage_tag} (Model: {cfg.model_type}) ===\n")
        f.write(f"=== Adaptive Balancing Used ===\n\n")
        f.write(metrics_str)
        f.write("\n")
    
    elapsed_time = time.time() - start_time
    gpus = GPUtil.getGPUs()
    if gpus:
        print(f"⏱ Stage {stage_tag} training: {elapsed_time:.2f}s on {gpus[0].name}")
    else:
        print(f"⏱ Stage {stage_tag} training: {elapsed_time:.2f}s on CPU")

    return os.path.join(out_dir, "trained_params.msgpack")
    
#============================================================
# 11. Hydra entry point
#============================================================
@hydra.main(config_path="config/train", config_name="stable_tensor_config", version_base=None)
def main(cfg: DictConfig):
    stage_tag = str(cfg.stage_tag)
    print(f"=== Stage {stage_tag} ({cfg.model_type}) - Adaptive Mode ===")
    
    # Import your loader (ensure this function is correct in your utils!)
    results = load_and_normalize_stagewise_data_stable(
        cfg.model_type, "datafiles", [stage_tag],
        seed=cfg.seed, scaling_mode=cfg.data.scaling_mode
    )
    
    Xtr, Xv, Xt, Ytr, Yv, Yt, X_mean, X_std, Y_mean, Y_std = results[stage_tag]
    
    if cfg.transfer_checkpoint is None or cfg.transfer_checkpoint == "null":
       transfer_ckpt = None
    else:
       transfer_ckpt = cfg.transfer_checkpoint
    
    # We ignore the lambda from config for "fixed" weighting
    # but pass it as a dummy or initial guess if needed
    run_training_for_lambda(cfg, 1.0, Xtr, Xv, Xt,
                            Ytr, Yv, Yt, X_mean, X_std, Y_mean, Y_std,
                            stage_tag, transfer_ckpt_path=transfer_ckpt)

if __name__ == "__main__":
    main()