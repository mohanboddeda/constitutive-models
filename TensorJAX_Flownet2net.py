#!/usr/bin/env python3
"""
TensorJAX_Flownet2net.py
------------------------
Unified training script for FLOW-SPECIFIC data (Maxwell-B focus) using MINING STRATEGY.
Adapts TensorJAX_Flow.py to support:
- "Inverse Pyramid" Data Loading (Variable sample sizes per stage).
- Net2Net Expansion (Widen/Deepen) for Curriculum Learning.
- Robust Weight Transfer from 'maxwellflow' mined data.

Key Features:
1. Loads data via 'utils.pretrain_flow_mining'.
2. Checks for Net2Net expansion triggers at every stage.
3. Saves models to 'trained_models/maxwellflow'.

FIXED VERSION: FORCE-STRIPS INPUT DIMENSION (9) TO PREVENT GHOST LAYERS.
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
from flax.core.frozen_dict import freeze, unfreeze
from flax import serialization  # <--- CRITICAL IMPORT FOR SAVING
from omegaconf import DictConfig
import GPUtil
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Enable x64 for high-precision physics calculations
jax.config.update("jax_enable_x64", True)

# --- Import Custom Utilities (FLOW MINING Versions) ---
# Pre-train: Data Loading & Checkpointing
from utils.pretrain_flow_mining import (
    load_and_normalize_flow_data_replay_mining, 
    save_checkpoint, 
    load_checkpoint
)

# Post-train: Analysis & Plotting
from utils.posttrain_flow import (
    plot_all_losses,
    plot_dataset_predictions_summary,
    plot_global_stress_summary
)

# Net2Net Utility
from utils.net2netflow import apply_net2net

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
    # Assuming steady state (dT/dt=0, v.grad=0) -> -L*T - T*L.T
    
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
    use_layernorm: bool = False

    @nn.compact
    def __call__(self, x, train=True):
        # Flatten input if necessary
        if x.ndim == 3: x = x.reshape((x.shape[0], -1))
        
        act_fn = self.activation_fn or nn.relu
        
        # Iterate through HIDDEN layers
        # If features=[128, 128, 128, 6], this loops 3 times.
        for i, feat in enumerate(self.features[:-1]):
            x = nn.Dense(feat, name=f'Dense_{i}')(x)
            if self.use_layernorm:
                x = nn.LayerNorm(name=f'LayerNorm_{i}')(x)
            x = act_fn(x)
            if self.dropout > 0:
                x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
        
        # Output layer
        # The name MUST be consecutive. If hidden were 0,1,2, output is 3.
        last_idx = len(self.features) - 1
        return nn.Dense(self.features[-1], name=f'Dense_{last_idx}')(x)

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
    # Always compute physics loss for logging, even if lambda is 0
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
        return init_lr * 0.5 * (1 + jnp.cos(jnp.pi * step / T_max_steps))
    return schedule_fn

#==============================================================
# 8. CRITICAL HELPER: Robust Weight Extraction
#==============================================================
def find_dense_layers(params_dict):
    """
    Recursively hunts for the dictionary level that contains 'Dense_0', 'Dense_1', etc.
    It peels away any 'params' or other wrapper keys until it finds the actual weights.
    This fixes the 'ScopeCollectionNotFound' error by ensuring we only pass raw weights to Net2Net.
    """
    # 1. Unfreeze to ensure we are working with standard dict
    curr = unfreeze(params_dict) if hasattr(params_dict, "unfreeze") else params_dict
    keys = list(curr.keys())
    
    # 2. Success Condition: Found a key starting with "Dense"
    if any(k.startswith("Dense") for k in keys):
        return curr
        
    # 3. Recursive Step: Check inside 'params'
    if "params" in keys:
        return find_dense_layers(curr["params"])
        
    # 4. Fallback
    return curr

#============================================================
# 9. Main Training Function (Per Stage, Per Flow)
#============================================================
def run_training_stage(cfg, flow_type, stage_tag, data_tuple, output_dir, transfer_params=None):
    
    stage_start_time = time.time()
    X_train, X_val, X_test, Y_train, Y_val, Y_test, X_mean, X_std, Y_mean, Y_std = data_tuple
    
    clean_flow = flow_type.replace("_", " ").title()
    print(f"\n🚀 Training Stage: {stage_tag} ({clean_flow}) | Training Samples: {X_train.shape[0]}")
    
    # 1. Folder Setup
    # Path: trained_models/maxwellflow/.../flow_type/maxwell_B_STAGE
    stage_dir = os.path.join(output_dir, f"{cfg.model_type}_{stage_tag}")
    fig_dir = os.path.join(stage_dir, "figures")
    os.makedirs(stage_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # 2. Residual Function
    if cfg.model_type == "maxwell_B":
        residual_fn = maxwellB_residual
    else:
        print(f"[WARN] Defaulting to Maxwell-B residual for {cfg.model_type}")
        residual_fn = maxwellB_residual

    # 3. Model Setup (FIXED: Handles Input Dim)
    # The config 'layers' usually is [Input, Hidden1, ..., Output]
    # We must slice [1:] to get [Hidden1, ..., Output] for the MLP constructor
    # because MLP automatically infers the input dimension.
    
    raw_layers = list(cfg.model.layers)
    
    # --- AUTO-FIX: If first element is small (likely input dim), strip it. ---
    # Assumption: Hidden layers are usually >= 32. Input is 9.
    if len(raw_layers) > 1 and raw_layers[0] < 32: 
        model_layers = raw_layers[1:]
        print(f"   [Auto-Fix] Stripped input dimension ({raw_layers[0]}) from config.")
    else:
        model_layers = raw_layers
        
    model_layers[-1] = 6 
    print(f"   [Model] Building MLP with layers: {model_layers}")
    
    # Check for LayerNorm setting (Default False to preserve behavior unless requested)
    use_ln = cfg.model.get("use_layernorm", False)
    if use_ln:
        print(f"   🛡️ Layer Normalization enabled.")

    act_fn = activation_map.get(cfg.model.activation, nn.relu)
    model = MLP(features=model_layers, dropout=cfg.model.dropout, activation_fn=act_fn, use_layernorm=use_ln)
    
    key = jax.random.PRNGKey(cfg.seed)
    dummy_input = jnp.ones((1, X_train.shape[1]))
    
    # 4. Initialization / Transfer / Net2Net Logic
    # ---------------------------------------------------------
    # First, initialize random weights to establish model structure/shapes
    params = model.init(key, dummy_input)
    
    raw_input_weights = None
    
    # PRIORITY 1: Memory Transfer (Continuous Multi-Stage Loop)
    if transfer_params is not None:
        print(f"   🔄 Memory Transfer: Initializing with weights from previous stage loop.")
        raw_input_weights = transfer_params
        
    # PRIORITY 2: File Transfer (Manual Multi-Stage OR Single Stage Fine-tuning)
    elif cfg.transfer_checkpoint:
        print(f"   🔄 File Transfer: Loading checkpoint from: {cfg.transfer_checkpoint}")
        
        # Template structure to load data into
        init_structure = {
            "params": None,  
            "X_mean": X_mean, "X_std": X_std, 
            "Y_mean": Y_mean, "Y_std": Y_std
        }
        
        try:
            restored = load_checkpoint(cfg.transfer_checkpoint, init_structure)
            if "params" in restored:
                print("   ✅ Weights successfully loaded from file.")
                raw_input_weights = restored["params"]
            else:
                print("   ⚠️ 'params' key missing in checkpoint root. Assuming direct weight dictionary.")
                raw_input_weights = restored
                
        except Exception as e:
            print(f"   ❌ Failed to load checkpoint: {e}. Falling back to random init.")

    # --- NET2NET EXPANSION LOGIC ---
    if raw_input_weights is not None:
        print("   ⚡ Checking for Net2Net expansion...")
        
        # A. Find the actual layers (Peel the onion)
        clean_weights = find_dense_layers(raw_input_weights)
        
        # B. Apply Net2Net (Widen or Deepen)
        expanded_weights = apply_net2net(clean_weights, cfg.model.layers)
        
        # C. Re-Wrap for Flax
        params = {"params": expanded_weights}
        print("      [Info] Model architecture adapted via Net2Net.")
        
    else:
        print("   🆕 Scratch: Starting from random initialization.")

    # 5. Optimizer & Scheduler
    # Use 'finetune_epochs' if transferring, else 'num_epochs'
    num_epochs = cfg.training.get("finetune_epochs", cfg.training.num_epochs) if raw_input_weights is not None else cfg.training.num_epochs
    
    steps_per_epoch = max(1, int(np.ceil(X_train.shape[0] / cfg.training.batch_size)))
    lr_schedule = cosine_annealing_lr(cfg.training.learning_rate, num_epochs, steps_per_epoch)
    optimizer = optax.adamw(learning_rate=lr_schedule, weight_decay=cfg.training.weight_decay)
    opt_state = optimizer.init(params)

    if cfg.model.dropout == 0.0 and cfg.training.weight_decay == 0.0 and not use_ln:
         print("   ⚠️ [Advice] No regularization detected. Consider ++model.dropout=0.1")

    # 6. Variables
    train_losses, val_losses = [], []
    train_d_losses, val_d_losses = [], []
    train_p_losses, val_p_losses = [], []
    best_val_loss = float('inf')
    target_lambda = cfg.training.lambda_phys

    # Default values
    avg_train_loss = 0.0; avg_train_d = 0.0; avg_train_p = 0.0
    val_total = 0.0; val_d = 0.0; val_p = 0.0

    if raw_input_weights is not None:
        print(f"   🛡️ Replay Mode: Physics constraints enabled fully (λ={target_lambda}).")
        print(f"   ⏳ Adapting for {num_epochs} epochs.")
    else:
        print(f"   📈 Curriculum Mode: Ramping physics constraints from 0.0 to {target_lambda} over {num_epochs} epochs.")

    # --- EPOCH LOOP ---
    for epoch in range(num_epochs):
        
        # Constant λ if Transfer, Ramp if Scratch
        if raw_input_weights is not None:
            lambda_curr = target_lambda
        else:
            lambda_curr = target_lambda * (epoch / num_epochs)

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
            
            params, opt_state, loss_val, d_loss, p_loss = train_step(params, opt_state, xb, yb, subkey)
            
            batch_size = e - s
            ep_total += loss_val.item() * batch_size
            ep_d += d_loss.item() * batch_size
            ep_p += p_loss.item() * batch_size

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
            
            # CRITICAL FIX FOR SERIALIZATION
            ckpt_params = serialization.to_state_dict(params)
            
            save_checkpoint(ckpt_params, X_mean, X_std, Y_mean, Y_std, os.path.join(stage_dir, "best_checkpoint.msgpack"))

        # Logging
        if (epoch + 1) % 50 == 0 or epoch == num_epochs - 1:
            avg_train_pure = avg_train_d + lambda_curr * avg_train_p
            print(f"   Ep {epoch+1:04d} | λ={lambda_curr:.3f} | Train: {avg_train_loss:.2e} (Pure:{avg_train_pure:.2e} D:{avg_train_d:.2e} P:{avg_train_p:.2e}) | Val: {val_total:.2e}")

    # 7. PLOTTING AND EVALUATION
    print(f"   📊 Generating Analysis Plots ({clean_flow})...")
    
    # Load Best Param
    restored = load_checkpoint(os.path.join(stage_dir, "best_checkpoint.msgpack"),
                               {"params": None, "X_mean": X_mean, "X_std": X_std, "Y_mean": Y_mean, "Y_std": Y_std})
    
    if "params" in restored:
        best_params = restored["params"]
        best_params = freeze(best_params)
    else:
        best_params = params

    # =========================================================
    # NEW: SAVE RAW LOGS TO DISK (Safety Net)
    # =========================================================
    history_path = os.path.join(stage_dir, "loss_history.npz")
    np.savez(
        history_path,
        train_d=train_d_losses,
        val_d=val_d_losses,
        train_p=train_p_losses,
        val_p=val_p_losses,
        Y_std=Y_std,            # Save scaling factor just in case
        n_samples=cfg.n_samples,
        flow_type=flow_type,
        stage_tag=stage_tag,
        model_type=cfg.model_type
    )
    print(f"   💾 Loss history saved to: {history_path}")
    # =========================================================

    if num_epochs > 0:
        plot_all_losses(train_d_losses, val_d_losses, 
                        train_p_losses, val_p_losses, 
                        Y_std, fig_dir, cfg.model_type, stage_tag,
                        flow_type=flow_type,
                        n_samples=cfg.n_samples)
    else:
        print("   ⚠️ Skipping Loss Plot (num_epochs=0).")

    # Denormalize Data (Physical Units)
    y_true_phys = np.array(Y_test) * np.array(Y_std) + np.array(Y_mean)
    y_pred_phys = np.array(model.apply(best_params, X_test, train=False)) * np.array(Y_std) + np.array(Y_mean)

    # --- 1. Global Stress Summary (The 4-panel Heatmap) ---
    try:
        plot_global_stress_summary(y_true_phys, y_pred_phys, fig_dir, cfg.model_type, flow_type=flow_type)
    except Exception as e:
        print(f"⚠️ Global Summary plot failed: {e}")

    # 2. Test Metrics
    test_total_loss, (test_d_loss, test_p_loss) = compute_losses(
        best_params, model, X_test, Y_test, cfg.training.lambda_phys, False, key,
        X_mean, X_std, Y_mean, Y_std, residual_fn, cfg.eta0, cfg.lam
    )
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
    shared_metrics_dir = stage_dir 
    my_log_name = f"metrics.txt"
    
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
        flow_type=flow_type,
        elapsed_time=elapsed
    )
    
    device = "GPU" if GPUtil.getGPUs() else "CPU"
    print(f"✅ Stage {stage_tag} Finished in {elapsed:.2f}s on {device}.")
    
    return best_params

#============================================================
# 9. Hydra Entry Point (Iterates over Flow Types)
#============================================================
@hydra.main(config_path="config/train", config_name="flow_net2net_config", version_base=None)
def main(cfg: DictConfig):
    
    total_start_time = time.time()

    print(f"\n{'='*60}")
    print(f"🚀 STARTING FLOW TRAINING (MINING+NET2NET) | Mode: {cfg.mode.upper()} | Model: {cfg.model_type}")
    print(f"🌊 Flows to Train: {cfg.flow_types}")
    print(f"{'='*60}\n")

    # Define Size Folder String (Generic for base path)
    if cfg.n_samples >= 1000:
       size_folder = f"{int(cfg.n_samples/1000)}ksamples"
    else:
       size_folder = f"{cfg.n_samples}samples"

    # --- MAIN LOOP: ITERATE OVER FLOW TYPES ---
    for flow_type in cfg.flow_types:
        try:
            print(f"\n >>> 🏁 PROCESSING FLOW: {flow_type.upper()} <<<")

            # 1. Output Path: trained_models/maxwellflow/{mode}/seed_XX/{flow_type}/{size_folder}
            if cfg.mode in ["single_stage", "multi_stage"]:
                base_out = os.path.join("trained_models", "maxwellflow", cfg.mode, f"seed_{cfg.seed}", flow_type, size_folder)
            else:
                raise ValueError(f"Invalid mode: {cfg.mode}")
            
            os.makedirs(base_out, exist_ok=True)

            # 2. Load Data for this SPECIFIC Flow Type using MINING LOADER
            # Note: This function handles the stages internal to that flow type
            data_stages = load_and_normalize_flow_data_replay_mining(
                flow_type=flow_type,
                model_type=cfg.model_type,
                data_root="datafiles", 
                mode=cfg.mode,
                seed=cfg.seed,
                n_samples=cfg.n_samples,
                scaling_mode=cfg.data.scaling_mode,
                replay_ratio=cfg.data.replay_ratio
            )
            
            # 3. Train Stages
            current_params = None
            
            # Note: cfg.transfer_checkpoint is handled inside run_training_stage now.

            if not data_stages:
                print(f"⚠️ No valid data stages found for {flow_type}. Skipping.")
                continue

            for stage_name, data_tuple in data_stages.items():
                
                # --- UNIVERSAL FILTER LOGIC ---
                # 1. Get target stage from config (default to None if not passed)
                target_stage = getattr(cfg, "stage", None)
                
                # 2. If user specified a specific stage, SKIP everything else.
                if target_stage and target_stage != stage_name:
                    continue
                # ------------------------------

                trained_params = run_training_stage(
                    cfg, flow_type, stage_name, data_tuple, base_out, transfer_params=current_params
                )
                
                # If multi-stage, carry over weights
                if cfg.mode == "multi_stage":
                    current_params = trained_params

        except Exception as e:
            print(f"❌ CRITICAL ERROR training {flow_type}: {e}")
            import traceback
            traceback.print_exc()

    total_elapsed = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"🏁 ALL FLOW TRAININGS COMPLETED in {total_elapsed:.2f}s")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()