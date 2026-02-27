import os
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.optimize 
import jax.flatten_util
import flax.linen as nn
import optax
import time
import GPUtil
from omegaconf import DictConfig
import hydra
from tabulate import tabulate
from sklearn.metrics import mean_absolute_error

# Import Utils
from utils.data_utils_replay import (
    load_and_normalize_stagewise_data_stable, 
    save_checkpoint, load_checkpoint,
    plot_all_losses, plot_residual_hist, plot_residuals_vs_pred,
    plot_stress_tensor_comparison, plot_dataset_predictions_summary
)

# --- 1. Physics & Helpers ---
def vec6_to_sym3(vec):
    T = jnp.zeros((vec.shape[0], 3, 3))
    T = T.at[:, 0, 0].set(vec[:, 0]); T = T.at[:, 1, 1].set(vec[:, 1]); T = T.at[:, 2, 2].set(vec[:, 2])
    T = T.at[:, 0, 1].set(vec[:, 3]); T = T.at[:, 1, 0].set(vec[:, 3])
    T = T.at[:, 0, 2].set(vec[:, 4]); T = T.at[:, 2, 0].set(vec[:, 4])
    T = T.at[:, 1, 2].set(vec[:, 5]); T = T.at[:, 2, 1].set(vec[:, 5])
    return T

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

# --- 2. Model ---
activation_map = {"relu": nn.relu, "tanh": nn.tanh, "sigmoid": nn.sigmoid}

class MLP(nn.Module):
    features: list
    dropout: float = 0.0
    activation_fn: callable = None
    @nn.compact
    def __call__(self, x, train=True):
        if x.ndim == 3: x = x.reshape((x.shape[0], -1))
        act_fn = self.activation_fn or nn.relu
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = act_fn(x)
            x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
        return nn.Dense(self.features[-1])(x)

# --- 3. Losses ---
def compute_losses(params, model, x_norm, y_norm, lambda_phys, train, dropout_key,
                   X_mean, X_std, Y_mean, Y_std, residual_fn, eta0, lam, lam_r=1.0):
    
    use_dropout = (train and (dropout_key is not None))
    preds_norm = model.apply(params, x_norm, train=train, 
                             rngs={'dropout': dropout_key} if use_dropout else {})
    
    # Data Loss
    preds_phys = preds_norm * Y_std + Y_mean
    y_phys = y_norm * Y_std + Y_mean
    data_loss = jnp.mean((preds_phys - y_phys) ** 2)

    # Physics Loss
    # No "if lambda > 0" check here to avoid JAX Tracer errors
    L_phys = x_norm * X_std + X_mean
    T_phys = vec6_to_sym3(preds_phys)
    
    if residual_fn == maxwellB_residual:
        residuals = residual_fn(L_phys.reshape(-1, 3, 3), T_phys, eta0, lam)
    else:
        residuals = residual_fn(L_phys.reshape(-1, 3, 3), T_phys, eta0, lam, lam_r)
        
    physics_loss = jnp.mean(residuals ** 2)

    total_loss = data_loss + lambda_phys * physics_loss
    return total_loss, (data_loss, physics_loss)

# --- 4. Main Training ---
def run_training_lbfgs(cfg, lambda_val, X_tr, X_v, X_t, Y_tr, Y_v, Y_t,
                       X_m, X_s, Y_m, Y_s, stage_tag, transfer_ckpt=None):
    
    out_dir = os.path.join(cfg.output_dir, f"{cfg.model_type}_stage_{stage_tag}_LBFGS")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    if cfg.model_type == "maxwell_B": residual_fn = maxwellB_residual
    elif cfg.model_type == "oldroyd_B": residual_fn = oldroydB_residual
    elif cfg.model_type == "ptt_exponential": residual_fn = ptt_exponential_residual
    else: raise ValueError(f"Unsupported model_type: {cfg.model_type}")

    # Setup Model
    activation_fn = activation_map.get(cfg.model.activation, nn.relu)
    model = MLP(features=list(cfg.model.layers)[:-1] + [6], 
                dropout=cfg.model.dropout, 
                activation_fn=activation_fn)
    
    seed_val = int(cfg.seed) if cfg.seed is not None else 42
    key = jax.random.PRNGKey(seed_val)
    params = model.init(key, jnp.ones([1, X_tr.shape[1]]))

    # Load Weights if Transfer
    if transfer_ckpt:
        print(f"🔄 Loading AdamW weights for initialization...")
        restored = load_checkpoint(transfer_ckpt, {"params": params, "X_mean": X_m, "X_std": X_s, "Y_mean": Y_m, "Y_std": Y_s})
        params = restored["params"]
        lambda_curr = lambda_val
    else:
        print("🚀 Starting fresh (AdamW Warmup first)")
        lambda_curr = 0.0

    # --- PHASE 1: ADAMW ---
    print("\n=== Phase 1: AdamW Warm-up ===")
    warmup_epochs = int(cfg.training.num_epochs * 0.8) 
    steps_per_epoch = int(np.ceil(X_tr.shape[0] / cfg.training.batch_size))
    lr_schedule = optax.cosine_decay_schedule(cfg.training.learning_rate, decay_steps=warmup_epochs*steps_per_epoch)
    optimizer = optax.adamw(learning_rate=lr_schedule, weight_decay=cfg.training.weight_decay)
    opt_state = optimizer.init(params)

    # Logging lists for plots
    train_losses, val_losses, train_data, val_data, train_phys, val_phys = [], [], [], [], [], []

    @jax.jit
    def adam_step(params, opt_state, x, y, lam_val, key):
        (loss, (d_loss, p_loss)), grads = jax.value_and_grad(compute_losses, has_aux=True)(
            params, model, x, y, lam_val, True, key, X_m, X_s, Y_m, Y_s, residual_fn, cfg.eta0, cfg.lam, cfg.lam_r
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, d_loss, p_loss
    
    for epoch in range(warmup_epochs):
        if not transfer_ckpt: lambda_curr = lambda_val * (epoch / warmup_epochs)
            
        perm = np.random.permutation(X_tr.shape[0])
        ep_loss, ep_d, ep_p = 0, 0, 0
        
        for i in range(0, X_tr.shape[0], cfg.training.batch_size):
            batch_x = X_tr[perm[i:i+cfg.training.batch_size]]
            batch_y = Y_tr[perm[i:i+cfg.training.batch_size]]
            step_key = jax.random.fold_in(key, epoch)
            params, opt_state, loss, d_loss, p_loss = adam_step(params, opt_state, batch_x, batch_y, lambda_curr, step_key)
            
            ep_loss += loss.item()
            ep_d += d_loss.item()
            ep_p += p_loss.item()
            
        # Logging per epoch
        n_batches = steps_per_epoch
        train_losses.append(ep_loss / n_batches)
        train_data.append(ep_d / n_batches)
        train_phys.append(ep_p / n_batches)

        # Validation
        v_loss, (v_d, v_p) = compute_losses(params, model, X_v, Y_v, lambda_curr, False, None, X_m, X_s, Y_m, Y_s, residual_fn, cfg.eta0, cfg.lam, cfg.lam_r)
        val_losses.append(v_loss.item())
        val_data.append(v_d.item())
        val_phys.append(v_p.item())

        if epoch % 50 == 0:
            print(f"AdamW Epoch {epoch}: Val Loss = {v_loss:.2e}")

    # --- PHASE 2: L-BFGS ---
    print("\n=== Phase 2: L-BFGS Polishing ===")
    flat_params, unflatten_fn = jax.flatten_util.ravel_pytree(params)

    @jax.jit
    def loss_wrapper(flat_p):
        p = unflatten_fn(flat_p)
        val, _ = compute_losses(p, model, X_tr, Y_tr, lambda_val, False, None, 
                                X_m, X_s, Y_m, Y_s, residual_fn, cfg.eta0, cfg.lam, cfg.lam_r)
        return val

    try:
        solver_result = jax.scipy.optimize.minimize(loss_wrapper, flat_params, method='BFGS', options={'maxiter': 500})
        params = unflatten_fn(solver_result.x)
        print(f"✅ L-BFGS Finished! Final Loss: {solver_result.fun:.2e}")
    except Exception as e:
        print(f"⚠️ L-BFGS Failed/Skipped: {e}")

    # --- FINAL PLOTTING & SAVING (The Missing Part) ---
    save_checkpoint(params, X_m, X_s, Y_m, Y_s, os.path.join(out_dir, "trained_params.msgpack"))
    
    # 1. Plot Training History (AdamW Phase)
    plot_all_losses(train_losses, val_losses, train_data, val_data, train_phys, val_phys, Y_s, fig_dir, cfg.model_type)

    # 2. Get Predictions
    y_true_phys = np.array(Y_t) * np.array(Y_s) + np.array(Y_m)
    y_pred_phys = np.array(model.apply(params, X_t, train=False)) * np.array(Y_s) + np.array(Y_m)

    # 3. Plot Physics Verification
    plot_stress_tensor_comparison(vec6_to_sym3, y_true_phys, y_pred_phys, [0, 5, 10], fig_dir, cfg.model_type)
    plot_dataset_predictions_summary(vec6_to_sym3, y_true_phys, y_pred_phys, fig_dir, cfg.model_type)
    
    # 4. Residual Plots
    residuals = y_true_phys - y_pred_phys
    plot_residual_hist(residuals, fig_dir, cfg.model_type)
    plot_residuals_vs_pred(y_pred_phys, residuals, fig_dir, cfg.model_type)

    # 5. Metrics Table
    test_loss, (d_loss, p_loss) = compute_losses(params, model, X_t, Y_t, lambda_val, False, None, X_m, X_s, Y_m, Y_s, residual_fn, cfg.eta0, cfg.lam, cfg.lam_r)
    
    metrics_table = [
       ["Test Total Loss", float(test_loss)],
       ["Test Data Loss", float(d_loss)],
       ["Test Phys Loss", float(p_loss)],
       ["Test MSE", float(np.mean(residuals ** 2))],
       ["Test MAE", mean_absolute_error(y_true_phys, y_pred_phys)]
    ]
    
    print("\nFinal Metrics:")
    print(tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="grid"))
    
    metrics_path = os.path.join(out_dir, f"metrics_stage_{stage_tag}.txt")
    with open(metrics_path, "w") as f:
        f.write(f"=== Metrics for Stage: {stage_tag} (L-BFGS) ===\n\n")
        f.write(tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="grid"))

    return os.path.join(out_dir, "trained_params.msgpack")

@hydra.main(config_path="config/train", config_name="stable_tensor_config", version_base=None)
def main(cfg: DictConfig):
    results = load_and_normalize_stagewise_data_stable(
        cfg.model_type, "datafiles", [str(cfg.stage_tag)], 
        seed=cfg.seed, scaling_mode=cfg.data.scaling_mode, replay_ratio=0.2
    )
    Xtr, Xv, Xt, Ytr, Yv, Yt, Xm, Xs, Ym, Ys = results[str(cfg.stage_tag)]
    ckpt = cfg.transfer_checkpoint if (cfg.transfer_checkpoint and cfg.transfer_checkpoint != "null") else None
    run_training_lbfgs(cfg, cfg.training.lambda_phys[0], Xtr, Xv, Xt, Ytr, Yv, Yt, Xm, Xs, Ym, Ys, str(cfg.stage_tag), ckpt)

if __name__ == "__main__":
    main()