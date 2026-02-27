"""
utils/net2netflow.py
--------------------
Implements "Net2Net" (Function Preserving Transformations) for JAX/Flax.
Reference: "Net2Net: Accelerating Learning via Knowledge Transfer" (Chen et al., ICLR 2016)

This module allows us to:
1. Widen layers (add neurons) by padding with zeros.
2. Deepen networks (add layers) by initializing as Identity.

Goal: Change the architecture 'capacity' without changing the model's function output.
Critical for the "Expansion" phase of the Multi-Stage Curriculum.
"""

import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import unfreeze, freeze

# =============================================================================
# 1. WIDEN LAYER (Net2WiderNet)
# =============================================================================
def widen_layer(params, layer_name, new_width, next_layer_name):
    """
    Performs the 'Net2WiderNet' transformation.
    Strategy: Zero-Padding (Preserves function exactly).
    """
    params = unfreeze(params)
    
    # Safety Check
    if layer_name not in params:
        return freeze(params)

    # Get Old Dimensions
    w_old = params[layer_name]['kernel'] # (In, Old_Width)
    b_old = params[layer_name]['bias']   # (Old_Width,)
    
    n_in, n_old = w_old.shape
    
    # If the layer is already big enough, don't touch it.
    if new_width <= n_old:
        return freeze(params)

    print(f"   [Net2Net] Widening {layer_name}: {n_old} -> {new_width} neurons.")

    # --- Step 1: Widen Current Layer ---
    # Create new larger containers initialized with Zeros
    w_new = jnp.zeros((n_in, new_width))
    b_new = jnp.zeros((new_width,))
    
    # Copy old weights into the "active" part. 
    # New neurons remain 0.0 (Dead Neurons initially).
    # FIX: Add small noise to new neurons to break symmetry and allow gradient flow.
    w_new = w_new.at[:, :n_old].set(w_old)
    b_new = b_new.at[:n_old].set(b_old)
    
    noise_scale = 1e-3
    new_cols = new_width - n_old
    
    if new_cols > 0:
        noise_w = np.random.normal(0.0, noise_scale, (n_in, new_cols))
        noise_b = np.random.normal(0.0, noise_scale, (new_cols,))
        w_new = w_new.at[:, n_old:].set(jnp.array(noise_w))
        b_new = b_new.at[n_old:].set(jnp.array(noise_b))
    
    params[layer_name]['kernel'] = w_new
    params[layer_name]['bias'] = b_new

    # --- Step 2: Adjust Next Layer (Input Alignment) ---
    # The next layer expects 'n_old' inputs. We must increase this to 'new_width'.
    if next_layer_name in params:
        w_next = params[next_layer_name]['kernel']
        n_next_out = w_next.shape[1]
        
        # Create new matrix with more ROWS (to accept more inputs)
        w_next_new = jnp.zeros((new_width, n_next_out))
        
        # Copy old weights to top rows. Bottom rows (new inputs) are 0.0.
        w_next_new = w_next_new.at[:n_old, :].set(w_next)
        
        # FIX: Add noise to outgoing weights for new neurons too
        if new_cols > 0:
            noise_next = np.random.normal(0.0, noise_scale, (new_cols, n_next_out))
            w_next_new = w_next_new.at[n_old:, :].set(jnp.array(noise_next))
        
        params[next_layer_name]['kernel'] = w_next_new
        
    return freeze(params)

# =============================================================================
# 2. DEEPEN NETWORK (Net2DeeperNet)
# =============================================================================
def deepen_network(params, new_layer_name, target_width):
    """
    Performs the 'Net2DeeperNet' transformation.
    Inserts a new Dense layer that acts as an Identity Matrix (I).
    """
    params = unfreeze(params)
    print(f"   [Net2Net] Deepening: Inserting {new_layer_name} (Identity Init).")
    
    w_identity = jnp.eye(target_width)
    b_zero = jnp.zeros((target_width,))
    
    params[new_layer_name] = {'kernel': w_identity, 'bias': b_zero}
    return freeze(params)

# =============================================================================
# 3. SAFETY: ALIGN OUTPUT LAYER
# =============================================================================
def align_output_input(params, output_layer_name, expected_input_dim):
    """
    Ensures the Output Layer accepts the correct input size from the last hidden layer.
    """
    params = unfreeze(params)
    if output_layer_name not in params: 
        return freeze(params)

    w_out = params[output_layer_name]['kernel']
    current_in, n_outputs = w_out.shape

    if current_in < expected_input_dim:
        print(f"   [Net2Net] Aligning Output Layer Input: {current_in} -> {expected_input_dim}")
        w_new = jnp.zeros((expected_input_dim, n_outputs))
        w_new = w_new.at[:current_in, :].set(w_out)
        params[output_layer_name]['kernel'] = w_new

    return freeze(params)

# =============================================================================
# 4. MASTER CONTROLLER
# =============================================================================
def apply_net2net(current_params, target_config_layers):
    """
    Master Controller.
    """
    # --- A. Clean Target List ---
    # Strip Input Dimension if present (e.g. [9, 128...] -> [128...])
    # This aligns the target list with the ACTUAL hidden layers we want.
    real_target_layers = list(target_config_layers)
    if len(real_target_layers) >= 5 and real_target_layers[0] < 32:
        # User passed [9, 128, 128, 128, 6]. We want [128, 128, 128, 6].
        real_target_layers = real_target_layers[1:]

    # --- B. Analyze Current Loaded Architecture ---
    # FIX: Sort numerically (Dense_2 before Dense_10)
    def get_layer_index(name):
        return int(name.split('_')[-1]) if '_' in name and name.split('_')[-1].isdigit() else 9999

    layer_names = sorted(list(current_params.keys()), key=get_layer_index)
    hidden_layers = layer_names[:-1] 
    output_layer = layer_names[-1]
    
    current_widths = [current_params[l]['bias'].shape[0] for l in hidden_layers]
    
    # Target Hidden Widths (Slice [:-1] to remove Output)
    target_hidden_widths = real_target_layers[:-1]
    
    # --- C. WIDENING PHASE ---
    needs_widening = False
    limit = min(len(current_widths), len(target_hidden_widths))
    
    for i in range(limit):
        if current_widths[i] < target_hidden_widths[i]:
            needs_widening = True
            break
            
    if needs_widening:
        print(f"⚡ Net2Net Triggered: Widening Detected.")
        params = current_params
        for i, layer in enumerate(hidden_layers):
            if i >= len(target_hidden_widths): break 
            
            is_last_hidden = (i == len(hidden_layers) - 1)
            next_layer = output_layer if is_last_hidden else hidden_layers[i+1]
            
            params = widen_layer(params, layer, target_hidden_widths[i], next_layer)
        current_params = params

    # --- D. DEEPENING PHASE ---
    if len(current_widths) < len(target_hidden_widths):
        print(f"⚡ Net2Net Triggered: Deepening.")
        params = unfreeze(current_params)
        
        # Move Output Layer
        old_out_name = output_layer
        new_last_index = len(target_hidden_widths)
        new_out_name = f"Dense_{new_last_index}"
        
        print(f"      -> Moving Output Layer: {old_out_name} -> {new_out_name}")
        params[new_out_name] = params.pop(old_out_name)
        
        # Fill Gap with Identity
        start_idx = len(hidden_layers) 
        end_idx = new_last_index
        width = target_hidden_widths[0] 
        
        for i in range(start_idx, end_idx):
            new_layer = f"Dense_{i}"
            params = deepen_network(freeze(params), new_layer, width)
            params = unfreeze(params)

        current_params = freeze(params)
        output_layer = new_out_name 

    # --- E. FINAL ALIGNMENT CHECK ---
    if len(target_hidden_widths) > 0:
        last_hidden_width = target_hidden_widths[-1]
        current_params = align_output_input(current_params, output_layer, last_hidden_width)

    return current_params