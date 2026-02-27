#!/usr/bin/env python3
"""
TensorJAX_Uniform.py
--------------------
Unified training script for UNIFORM Grid data (Maxwell-B focus).
Mirrors the logic of TensorJAX_Random.py but adapted for:
- Uniform data directory structure.
- Uniform-specific pre/post-processing utilities.
- Robust Weight Transfer and Metrics Logging.
"""

#============================================================
# 0. Imports
#============================================================
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import hydra
from omegaconf import DictConfig
import GPUtil
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Enable x64 for high-precision physics calculations
jax.config.update("jax_enable_x64", True)

# --- Import Custom Utilities (UNIFORM Versions) ---
# Pre-train: Data Loading & Checkpointing
from utils.pretrain_uniform import (
    load_and_normalize_stagewise_data_replay, 
    save_checkpoint, 
    load_checkpoint
)

# Post-train: Analysis & Plotting
from utils.posttrain_uniform import (
    plot_all_losses,
    plot_dataset_predictions_summary,
    plot_global_stress_summary
)

#=================================================================
# 1. Helpers : convert 6-comp vector to 3x3 matrix (JAX version)
#=================================================================
def vec6_to_sym3_jax(vec):
    """
    Converts (N, 6) vector [xx, yy, zz, xy, xz, yz] -> (N, 3, 3) symmetric matrix.
    """
    N = vec.shape[0]
    T = jnp.zeros((N, 3, 3))
    # Diagonals
    T = T.at[:, 0, 0].set(vec[:, 0])
    T = T.at[:, 1, 1].set(vec[:, 1])
    T = T.at[:, 2, 2].set(vec[:, 2])
    # Off-diagonals (symmetric)
    T = T.at[:, 0, 1].set(vec[:, 3]); T = T.at[:, 1, 0].set(vec[:, 3])
    T = T.at[:, 0, 2].set(vec[:, 4]); T = T.at[:, 2, 0].set(vec[:, 4])
    T = T.at[:, 1, 2].set(vec[:, 5]); T = T.at[:, 2, 1].set(vec[:, 5])
    return T

#==============================================================
# 2. Physics based residuals (Maxwell-B)
#==============================================================
def maxwellB_residual(L_phys, T_phys, eta0, lam):
    """
    Computes the residual of the Maxwell-B Constitutive Equation.
    R = T + lambda * T_upper_convected - 2 * eta0 * D
    """
    # Rate of deformation tensor D = 0.5 * (L + L.T)
    D = 0.5 * (L_phys + jnp.swapaxes(L_phys, 1, 2))
    dim = L_phys.shape[1]
    I = jnp.eye(dim)
    
    # Upper Convected Derivative parts
    # T_upper = dT/dt + (v.grad)T - L*T - T*L.T
    # Here assuming steady state (dT/dt=0, v.grad=0) -> -L*T - T*L.T
    
    # Term A: (I - lam * L) * T
    A = I - lam * L_phys
    # Term B: T * (-lam * L.T)
    B = -lam * jnp.swapaxes(L_phys, 1, 2)
    # Term C: 2 * eta0 * D
    C = 2.0 * eta0 * D
    
    R = jnp.matmul(A, T_phys) + jnp.matmul(T_phys, B) - C
    return R

#==============================================================
# 3. Activation mapping
#==============================================================
activation_map = {
    "relu": nn.relu,
    "tanh": nn.tanh,
    "sigmoid": nn.sigmoid,
    "gelu": nn.gelu
}

#==============================================================
# 4. MLP Model
#==============================================================
class MLP(nn.Module):
    features: list
    dropout: float = 0.0
    activation_fn: callable = None

    @nn.compact
    def __call__(self, x, train=True):
        # Flatten input if necessary
        if x.ndim == 3: x = x.reshape((x.shape[0], -1))
        
        act_fn = self.activation_fn or nn.relu
        
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = act_fn(x)
            if self.dropout > 0:
                x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
        
        # Output layer (Linear)
        return nn.Dense(self.features[-1])(x)

#============================================================
# 5. Compute Data and Physics Losses 
#============================================================
def compute_losses(params, model, x_norm, y_norm, lambda_phys, train, rng_key,
                   X_mean, X_std, Y_mean, Y_std, residual_fn, eta0, lam):
    """
    Computes Total Loss = MSE_Data + lambda * MSE_Physics
    """
    # Forward Pass
    preds_norm = model.apply(params, x_norm, train=train, 
                             rngs={'dropout': rng_key} if train else {})
    
    # 1. Data Loss (Calculated in Physical Units for consistency)
    preds_phys = preds_norm * Y_std + Y_mean
    y_phys = y_norm * Y_std + Y_mean
    data_loss = jnp.mean((preds_phys - y_phys) ** 2)

    # 2. Physics Loss
    physics_loss = 0.0
    if lambda_phys > 0:
        L_phys = x_norm * X_std + X_mean
        T_phys = vec6_to_sym3_jax(preds_phys) 
        
        # Reshape L to (N, 3, 3)
        L_phys_3x3 = L_phys.reshape(-1, 3, 3)
        
        residuals = residual_fn(L_phys_3x3, T_phys, eta0, lam)
        physics_loss = jnp.mean(residuals ** 2)

    total_loss = data_loss + lambda_phys * physics_loss
    return total_loss, (data_loss, physics_loss)

#==============================================================
# 6. Training step function
#==============================================================
def make_train_step(model, optimizer, lambda_phys, X_mean, X_std, Y_mean, Y_std, residual_fn, eta0, lam):
    @jax.jit
    def train_step(params, opt_state, x, y, rng_key):
        loss_fn = lambda p: compute_losses(
            p, model, x, y, lambda_phys, True, rng_key,
            X_mean, X_std, Y_mean, Y_std, residual_fn, eta0, lam
        )
        (loss_val, (d_loss, p_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val, d_loss, p_loss
    return train_step

#==============================================================
# 7. Cosine LR schedule
#==============================================================
def cosine_annealing_lr(init_lr, T_max_epochs, steps_per_epoch):
    T_max_steps = T_max_epochs * steps_per_epoch
    def schedule_fn(step):
        # 0.5 * lr * (1 + cos(pi * step / total_steps))
        return init_lr * 0.5 * (1 + jnp.cos(jnp.pi * step / T_max_steps))
    return schedule_fn

#============================================================
# 8. Main training function (Per Stage)
#============================================================
def run_training_stage(cfg, stage_tag, data_tuple, output_dir, transfer_params=None):
    
    stage_start_time = time.time()
    X_train, X_val, X_test, Y_train, Y_val, Y_test, X_mean, X_std, Y_mean, Y_std = data_tuple
    
    print(f"\n🚀 Training Stage: {stage_tag} (Uniform) | Training Samples: {X_train.shape[0]}")
    
    # 1. Folder Setup
    # Path: trained_models/uniform/multi_stage/seed_XX/maxwell_B_STAGE
    stage_dir = os.path.join(output_dir, f"{cfg.model_type}_{stage_tag}")
    fig_dir = os.path.join(stage_dir, "figures")
    os.makedirs(stage_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # 2. Select Residual Function (Maxwell B)
    if cfg.model_type == "maxwell_B":
        residual_fn = maxwellB_residual
    else:
        print(f"[WARN] Defaulting to Maxwell-B residual for {cfg.model_type}")
        residual_fn = maxwellB_residual

    # 3. Model Setup
    model_layers = list(cfg.model.layers)
    model_layers[-1] = 6 # Force output dim to 6 (xx, yy, zz, xy, xz, yz)
    
    act_fn = activation_map.get(cfg.model.activation, nn.relu)
    model = MLP(features=model_layers, dropout=cfg.model.dropout, activation_fn=act_fn)
    
    key = jax.random.PRNGKey(cfg.seed)
    dummy_input = jnp.ones((1, X_train.shape[1]))
    
    # 4. Initialization / Transfer Logic
    # ---------------------------------------------------------
    # First, initialize random weights to establish model structure/shapes
    params = model.init(key, dummy_input)
    
    # PRIORITY 1: Memory Transfer (Continuous Multi-Stage Loop)
    if transfer_params is not None:
        print(f"   🔄 Memory Transfer: Initializing with weights from previous stage loop.")
        params = transfer_params
        
    # PRIORITY 2: File Transfer (Manual Multi-Stage OR Single Stage Fine-tuning)
    elif cfg.transfer_checkpoint:
        print(f"   🔄 File Transfer: Loading checkpoint from: {cfg.transfer_checkpoint}")
        
        # Template structure to load data into
        init_structure = {
            "params": params, 
            "X_mean": X_mean, "X_std": X_std, 
            "Y_mean": Y_mean, "Y_std": Y_std
        }
        
        try:
            restored = load_checkpoint(cfg.transfer_checkpoint, init_structure)
            if "params" in restored:
                params = restored["params"]
                print("   ✅ Weights successfully loaded from file.")
            else:
                print("   ⚠️ Checkpoint loaded, but 'params' missing! Using random init.")
        except Exception as e:
            print(f"   ❌ Error loading checkpoint: {e}")
            print("   -> Continuing with random initialization...")

    # PRIORITY 3: Scratch (Standard Single Stage / First Stage)
    else:
        print("   🆕 Scratch: Starting from random initialization.")

    # 5. Optimizer & Scheduler
    steps_per_epoch = max(1, int(np.ceil(X_train.shape[0] / cfg.training.batch_size)))
    lr_schedule = cosine_annealing_lr(cfg.training.learning_rate, cfg.training.num_epochs, steps_per_epoch)
    optimizer = optax.adamw(learning_rate=lr_schedule, weight_decay=cfg.training.weight_decay)
    opt_state = optimizer.init(params)

    # 6. Training Loop Variables
    train_losses, val_losses = [], []
    train_d_losses, val_d_losses = [], []
    train_p_losses, val_p_losses = [], []
    best_val_loss = float('inf')
    
    target_lambda = cfg.training.lambda_phys

    # Default values in case num_epochs=0 (Plotting Mode)
    avg_train_loss = 0.0
    avg_train_d = 0.0
    avg_train_p = 0.0
    val_total = 0.0
    val_d = 0.0
    val_p = 0.0

    # Determine initial mode for print statement
    if transfer_params is not None or cfg.transfer_checkpoint:
        print(f"   🛡️ Replay Mode: Physics constraints enabled fully (λ={target_lambda}).")
    else:
        print(f"   📈 Curriculum Mode: Ramping physics constraints from 0.0 to {target_lambda} over {cfg.training.num_epochs} epochs.")

    # --- EPOCH LOOP ---
    for epoch in range(cfg.training.num_epochs):
        
        # --- FIXED LOGIC: Constant λ if Transfer, Ramp if Scratch ---
        if transfer_params is not None or cfg.transfer_checkpoint:
            # Transfer Mode: Constant Lambda
            lambda_curr = target_lambda
        else:
            # Scratch Mode: Linear Ramp
            lambda_curr = target_lambda * (epoch / cfg.training.num_epochs)

        # Prepare Step Function
        train_step = make_train_step(model, optimizer, lambda_curr, 
                                     X_mean, X_std, Y_mean, Y_std, residual_fn, cfg.eta0, cfg.lam)
        
        # Shuffle
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(key, X_train.shape[0])
        X_sh, Y_sh = X_train[perm], Y_train[perm]
        
        ep_total, ep_d, ep_p = 0, 0, 0

        # Batch Loop
        for i in range(steps_per_epoch):
            s = i * cfg.training.batch_size
            e = min((i + 1) * cfg.training.batch_size, X_train.shape[0])
            xb, yb = X_sh[s:e], Y_sh[s:e]
            
            # Update
            params, opt_state, loss_val, d_loss, p_loss = train_step(params, opt_state, xb, yb, subkey)
            
            batch_size = e - s
            ep_total += loss_val.item() * batch_size
            ep_d += d_loss.item() * batch_size
            ep_p += p_loss.item() * batch_size

        # Averages
        avg_train_loss = ep_total / X_train.shape[0]
        avg_train_d = ep_d / X_train.shape[0]
        avg_train_p = ep_p / X_train.shape[0]
        
        train_losses.append(avg_train_loss)
        train_d_losses.append(avg_train_d)
        train_p_losses.append(avg_train_p)

        # Validation
        val_total, (val_d, val_p) = compute_losses(
            params, model, X_val, Y_val, lambda_curr, False, key,
            X_mean, X_std, Y_mean, Y_std, residual_fn, cfg.eta0, cfg.lam
        )
        val_total = float(val_total)
        
        val_losses.append(val_total)
        val_d_losses.append(float(val_d))
        val_p_losses.append(float(val_p))

        # Checkpoint
        if val_total < best_val_loss:
            best_val_loss = val_total
            save_checkpoint(params, X_mean, X_std, Y_mean, Y_std, os.path.join(stage_dir, "best_checkpoint.msgpack"))

        # Logging
        if (epoch + 1) % 50 == 0 or epoch == cfg.training.num_epochs - 1:
            print(f"   Ep {epoch+1:04d} | λ={lambda_curr:.3f} | Train: {avg_train_loss:.2e} | Val: {val_total:.2e}")

    # 7. PLOTTING AND EVALUATION
    print(f"   📊 Generating Analysis Plots...")
    
    # Load Best Params
    restored = load_checkpoint(os.path.join(stage_dir, "best_checkpoint.msgpack"),
                               {"params": params, "X_mean": X_mean, "X_std": X_std, "Y_mean": Y_mean, "Y_std": Y_std})
    best_params = restored["params"]
    
    # Only plot losses if we actually trained
    if cfg.training.num_epochs > 0:
        plot_all_losses(train_d_losses, val_d_losses, 
                        train_p_losses, val_p_losses, 
                        Y_std, fig_dir, cfg.model_type, stage_tag,
                        n_samples=cfg.n_samples) 

    # Denormalize Data (Physical Units)
    y_true_phys = np.array(Y_test) * np.array(Y_std) + np.array(Y_mean)
    y_pred_phys = np.array(model.apply(best_params, X_test, train=False)) * np.array(Y_std) + np.array(Y_mean)

    # --- 1. Global Stress Summary (The 4-panel Heatmap) ---
    try:
        plot_global_stress_summary(y_true_phys, y_pred_phys, fig_dir, cfg.model_type)
    except Exception as e:
        print(f"⚠️ Global Summary plot failed: {e}")

    # 2. Test Metrics Calculation
    test_total_loss, (test_d_loss, test_p_loss) = compute_losses(
        best_params, model, X_test, Y_test, cfg.training.lambda_phys, False, key,
        X_mean, X_std, Y_mean, Y_std, residual_fn, cfg.eta0, cfg.lam
    )
    # Calculate Mean Squared Error manually on physical units
    residuals_phys = y_true_phys - y_pred_phys
    test_mse = float(np.mean(residuals_phys**2))
    test_mae = mean_absolute_error(y_true_phys, y_pred_phys)

    # 3. Metrics Table
    metrics_table = [
        ["Train/total_loss", avg_train_loss],
        ["Train/data_loss", avg_train_d],
        ["Train/physics_loss", avg_train_p],
        ["Val/total_loss", val_total],
        ["Val/data_loss", float(val_d)],
        ["Val/physics_loss", float(val_p)],
        ["Test/total_loss", float(test_total_loss)],
        ["Test/data_loss", float(test_d_loss)],
        ["Test/physics_loss", float(test_p_loss)],
        ["Test/MSE", test_mse],
        ["Test/MAE", test_mae]
    ]

    # 4. Save Summary & Table to SPECIFIC Run Folder
    # UPDATED: We now save the metrics text file INSIDE the stage-specific directory
    shared_metrics_dir = stage_dir 
    my_log_name = f"{cfg.model_type}_{stage_tag}_metrics.txt"

    # --- UPDATED: Calculate Time & Pass to Plotter ---
    elapsed = time.time() - stage_start_time
    
    plot_dataset_predictions_summary(
        y_true_phys, y_pred_phys, 
        fig_dir=fig_dir,               
        shared_log_dir=shared_metrics_dir, 
        model_type=cfg.model_type, 
        metrics_table=metrics_table, 
        seed=cfg.seed,
        log_filename=my_log_name,
        n_samples=cfg.n_samples,
        stage_tag=stage_tag,
        elapsed_time=elapsed # <--- PASSED TIME
    )
    
    device = "GPU" if GPUtil.getGPUs() else "CPU"
    print(f"✅ Stage {stage_tag} Finished in {elapsed:.2f}s on {device}.")
    
    return best_params

#============================================================
# 9. Hydra Entry Point
#============================================================
@hydra.main(config_path="config/train", config_name="uniform_tensor_config", version_base=None)
def main(cfg: DictConfig):
    
    total_start_time = time.time()

    # Define Global Output Directory
    if cfg.n_samples >= 1000:
       size_folder = f"{int(cfg.n_samples/1000)}ksamples"
    else:
       size_folder = f"{cfg.n_samples}samples"
    if cfg.mode in ["single_stage", "multi_stage"]:
        base_out = os.path.join("trained_models", "uniform", cfg.mode, f"seed_{cfg.seed}", size_folder)
    else:
        raise ValueError(f"Invalid mode: {cfg.mode}")
    
    os.makedirs(base_out, exist_ok=True)
    
    # Device Info
    device_info = GPUtil.getGPUs()
    gpu_name = device_info[0].name if device_info else "CPU"
    print(f"\n{'='*60}")
    print(f"🚀 STARTING TRAINING (UNIFORM) | Mode: {cfg.mode.upper()} | Model: {cfg.model_type}")
    print(f"📂 Output Dir: {base_out}")
    print(f"🖥️  Compute Device: {gpu_name}")
    print(f"{'='*60}\n")

    # Load All Data Stages (Uniform Loader)
    # Note: Using Replay Loader with ratio=0.2 (20% old data mixed in)
    data_stages = load_and_normalize_stagewise_data_replay(
        model_type=cfg.model_type,
        data_root="datafiles", 
        mode=cfg.mode,
        seed=cfg.seed,
        n_samples=cfg.n_samples,
        scaling_mode=cfg.data.scaling_mode,
        replay_ratio=cfg.data.replay_ratio
    )
    
    current_params = None
    
    # Note: cfg.transfer_checkpoint is handled inside run_training_stage now.

    # Iterate through stages
    for stage_name, data_tuple in data_stages.items():
        
        # --- UNIVERSAL FILTER LOGIC ---
        # 1. Get target stage from config (default to None if not passed)
        target_stage = getattr(cfg, "stage", None)
        
        # 2. If user specified a specific stage, SKIP everything else.
        if target_stage and target_stage != stage_name:
            continue
        # ------------------------------

        # Train current stage
        trained_params = run_training_stage(
            cfg, stage_name, data_tuple, base_out, transfer_params=current_params
        )
        
        # Update current_params for the next iteration (only matters if running continuous loop)
        current_params = trained_params

    total_elapsed = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"🏁 ALL STAGES COMPLETED in {total_elapsed:.2f}s")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()