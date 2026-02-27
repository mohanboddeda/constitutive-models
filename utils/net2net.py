"""
utils/net2net.py
----------------
Implements "Net2Net" (Function Preserving Transformations) for JAX/Flax.
Reference: "Net2Net: Accelerating Learning via Knowledge Transfer" (Chen et al., ICLR 2016)

This module allows us to:
1. Widen layers (add neurons) by padding with zeros.
2. Deepen networks (add layers) by initializing as Identity.

Goal: Change the architecture 'capacity' without changing the model's function output.
"""

# We only need jax.numpy (for array ops) and flax (for dictionary handling)
import jax.numpy as jnp
from flax.core.frozen_dict import unfreeze, freeze

def widen_layer(params, layer_name, new_width, next_layer_name):
    """
    Performs the 'Net2WiderNet' transformation on a specific layer.
    
    Strategy: Zero-Padding
    1. Expand the current layer's weight matrix with columns of Zeros.
    2. Expand the *next* layer's weight matrix with rows of Zeros.
    
    Result: The new neurons output 0, and the next layer ignores them. 
            Function output is strictly preserved.
    """
    # Flax parameters are immutable (FrozenDict). We must unfreeze to edit.
    params = unfreeze(params)
    
    # --- Step 1: Widen the Current Layer (Source) ---
    # Check if layer exists to avoid key errors
    if layer_name not in params:
        print(f"   [Net2Net] Warning: {layer_name} not found. Skipping.")
        return freeze(params)

    w_old = params[layer_name]['kernel'] # Shape: (Inputs, Old_Width)
    b_old = params[layer_name]['bias']   # Shape: (Old_Width,)
    
    n_in, n_old = w_old.shape
    
    # Sanity check: If we are already wider or equal, don't do anything.
    if new_width <= n_old:
        return freeze(params)

    print(f"   [Net2Net] Widening {layer_name}: {n_old} -> {new_width} neurons.")

    # Create new container (JAX arrays are immutable, so we create fresh zeros)
    w_new = jnp.zeros((n_in, new_width))
    b_new = jnp.zeros((new_width,))
    
    # Copy old weights into the top-left (or beginning)
    # The rest remains 0.0 (Dead Neurons)
    w_new = w_new.at[:, :n_old].set(w_old)
    b_new = b_new.at[:n_old].set(b_old)
    
    # Save back to params
    params[layer_name]['kernel'] = w_new
    params[layer_name]['bias'] = b_new

    # --- Step 2: Adjust the Next Layer (Target) ---
    # The next layer expects input size = Old_Width. 
    # We must increase its input size to New_Width to match the previous layer.
    if next_layer_name in params:
        w_next = params[next_layer_name]['kernel'] # Shape: (Old_Width, Next_Output)
        n_next_out = w_next.shape[1]
        
        # Create new matrix with more ROWS (to accept more inputs)
        w_next_new = jnp.zeros((new_width, n_next_out))
        
        # Copy old weights to top rows. 
        # The bottom rows (corresponding to new neurons) are 0.0.
        # This ensures: Output = (Old_Input * Old_Weight) + (New_Input * 0.0)
        w_next_new = w_next_new.at[:n_old, :].set(w_next)
        
        params[next_layer_name]['kernel'] = w_next_new
        
    return freeze(params)

def deepen_network(params, new_layer_name, target_width):
    """
    Performs the 'Net2DeeperNet' transformation.
    
    Strategy: Identity Initialization
    Inserts a new Dense layer that acts as an Identity Matrix (I).
    Output = Input * I + 0 = Input.
    
    Result: The network gets deeper, but the signal flows through unchanged initially.
    """
    params = unfreeze(params)
    print(f"   [Net2Net] Deepening: Inserting {new_layer_name} (Identity Init).")
    
    # 1. Weights: Identity Matrix
    # We assume the layer connects neurons of size 'target_width' to 'target_width'
    w_identity = jnp.eye(target_width)
    
    # 2. Bias: Zeros
    b_zero = jnp.zeros((target_width,))
    
    # Create the new layer entry
    params[new_layer_name] = {
        'kernel': w_identity,
        'bias': b_zero
    }
    
    return freeze(params)

def apply_net2net(current_params, target_config_layers):
    """
    Master Controller.
    Compares the loaded parameter shapes against the target configuration
    and applies Widening or Deepening as needed.
    
    Args:
        current_params: Dictionary of trained weights (from checkpoint)
        target_config_layers: List from config, e.g., [128, 128, 128, 6]
    """
    # 1. Analyze Current Architecture
    # Flax layers are usually named 'Dense_0', 'Dense_1', etc.
    layer_names = sorted(list(current_params.keys())) 
    
    # Separate Hidden Layers vs Output Layer (based on assumption last one is output)
    hidden_layers = layer_names[:-1] 
    output_layer = layer_names[-1]
    
    # Detect widths from Bias shapes (easier than Kernel shapes)
    # bias shape is (width,)
    current_widths = [current_params[l]['bias'].shape[0] for l in hidden_layers]
    
    # Target widths from config (excluding the last output dimension 6)
    target_widths = list(target_config_layers)[:-1] 
    
    # --- LOGIC 1: Check for WIDENING (Improved Loop) ---
    # We loop through ALL layers to check if ANY need widening.
    needs_widening = False
    
    # Ensure lists are comparable length (handle deepening later)
    limit = min(len(current_widths), len(target_widths))
    
    for i in range(limit):
        if current_widths[i] < target_widths[i]:
            needs_widening = True
            break
            
    if needs_widening:
        print(f"⚡ Net2Net Triggered: Widening Detected.")
        params = current_params
        
        # Iterate through all hidden layers and widen them
        # Note: We iterate up to len(current_widths) because we can only widen existing layers
        for i, layer in enumerate(hidden_layers):
            if i >= len(target_widths): break # Safety check
            
            # Determine who is the 'next' layer to adjust its inputs
            # If we are at the last hidden layer, the 'next' is the Output Layer
            is_last_hidden = (i == len(hidden_layers) - 1)
            next_layer = output_layer if is_last_hidden else hidden_layers[i+1]
            
            # Apply transformation
            params = widen_layer(params, layer, target_widths[i], next_layer)
            
        current_params = params

    # --- LOGIC 2: Check for DEEPENING ---
    # If loaded model has fewer layers than target config
    if len(current_widths) < len(target_widths):
        print(f"⚡ Net2Net Triggered: Deepening from {len(current_widths)} layers to {len(target_widths)} layers")
        
        params = unfreeze(current_params)
        
        # Strategy: Insert new layers just before the Output Layer.
        
        # 1. Rename the current Output Layer to the new last index
        old_out_name = output_layer
        new_last_index = len(target_config_layers) - 1
        new_out_name = f"Dense_{new_last_index}"
        
        print(f"      -> Moving Output Layer: {old_out_name} -> {new_out_name}")
        params[new_out_name] = params.pop(old_out_name)
        
        # 2. Fill the gap with Identity Layers
        # If we had Dense_0, Dense_1... and we moved Out to Dense_3.
        # We need to fill Dense_2.
        start_idx = len(hidden_layers) 
        end_idx = new_last_index
        width = target_widths[0] # Assume uniform width for new layers
        
        for i in range(start_idx, end_idx):
            new_layer = f"Dense_{i}"
            params = deepen_network(freeze(params), new_layer, width)
            # Must unfreeze again for next loop iteration
            params = unfreeze(params)

        current_params = freeze(params)

    return current_params