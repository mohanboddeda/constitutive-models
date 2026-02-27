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

FIXED VERSION: 
- Correctly aligns hidden layers (Slicing [1:-1]).
- Safety Check: Ensures Output Layer input dimension matches the last Hidden Layer.
"""

import jax.numpy as jnp
from flax.core.frozen_dict import unfreeze, freeze

# =============================================================================
# 1. WIDEN LAYER (Net2WiderNet)
# =============================================================================
def widen_layer(params, layer_name, new_width, next_layer_name):
    """
    Performs the 'Net2WiderNet' transformation on a specific layer.
    
    Strategy: Zero-Padding
    1. Expand the current layer's weight matrix (cols) and bias with Zeros.
    2. Expand the *next* layer's weight matrix (rows) with Zeros.
    
    Args:
        params: Mutable dictionary of model parameters.
        layer_name: The name of the layer to widen (e.g., 'Dense_0').
        new_width: The target number of neurons (e.g., 128).
        next_layer_name: The name of the subsequent layer (e.g., 'Dense_1').
    """
    # Flax parameters are immutable (FrozenDict). We must unfreeze to edit.
    params = unfreeze(params)
    
    # --- Check for existence ---
    if layer_name not in params:
        print(f"   [Net2Net] Warning: {layer_name} not found. Skipping.")
        return freeze(params)

    # --- Get Old Dimensions ---
    w_old = params[layer_name]['kernel'] # Shape: (Inputs, Old_Width)
    b_old = params[layer_name]['bias']   # Shape: (Old_Width,)
    
    n_in, n_old = w_old.shape
    
    # --- Sanity Check ---
    # If the layer is already big enough, don't touch it.
    if new_width <= n_old:
        return freeze(params)

    print(f"   [Net2Net] Widening {layer_name}: {n_old} -> {new_width} neurons.")

    # --- Step 1: Widen Current Layer ---
    # Create new larger containers initialized with Zeros
    w_new = jnp.zeros((n_in, new_width))
    b_new = jnp.zeros((new_width,))
    
    # Copy old weights into the "active" part (indices 0 to n_old)
    # The new neurons (indices n_old to new_width) remain 0.0 (Dead Neurons initially)
    w_new = w_new.at[:, :n_old].set(w_old)
    b_new = b_new.at[:n_old].set(b_old)
    
    # Save back to params
    params[layer_name]['kernel'] = w_new
    params[layer_name]['bias'] = b_new

    # --- Step 2: Adjust Next Layer (Input Alignment) ---
    # The next layer expects 'n_old' inputs. We must increase this to 'new_width'.
    # Because the new neurons output 0.0, the weights connecting them to the next layer
    # can be anything, but we set them to 0.0 to be safe and preserve exact function identity.
    
    if next_layer_name in params:
        w_next = params[next_layer_name]['kernel'] # Shape: (Old_Width, Next_Output)
        n_next_out = w_next.shape[1]
        
        # Create new matrix with more ROWS (to accept more inputs)
        w_next_new = jnp.zeros((new_width, n_next_out))
        
        # Copy old weights to top rows.
        # Bottom rows (corresponding to new neurons) are 0.0.
        w_next_new = w_next_new.at[:n_old, :].set(w_next)
        
        params[next_layer_name]['kernel'] = w_next_new
        
    return freeze(params)


# =============================================================================
# 2. DEEPEN NETWORK (Net2DeeperNet)
# =============================================================================
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


# =============================================================================
# 3. SAFETY: ALIGN OUTPUT LAYER
# =============================================================================
def align_output_input(params, output_layer_name, expected_input_dim):
    """
    Ensures the Output Layer accepts the correct input size from the last hidden layer.
    This fixes the edge case where Deepening happens but the Output layer 
    still expects the old (small) input size.
    """
    params = unfreeze(params)
    
    # Safety Check: If Output layer doesn't exist, skip
    if output_layer_name not in params: 
        return freeze(params)

    w_out = params[output_layer_name]['kernel'] # Shape: (Current_In, Outputs)
    current_in, n_outputs = w_out.shape

    # If the input dimension is too small, we must expand it (padding with zeros)
    if current_in < expected_input_dim:
        print(f"   [Net2Net] Aligning Output Layer Input: {current_in} -> {expected_input_dim}")
        
        # Create new weight matrix with more rows
        w_new = jnp.zeros((expected_input_dim, n_outputs))
        
        # Copy old weights to the top
        w_new = w_new.at[:current_in, :].set(w_out)
        
        params[output_layer_name]['kernel'] = w_new

    return freeze(params)


# =============================================================================
# 4. MASTER CONTROLLER (Apply Logic)
# =============================================================================
def apply_net2net(current_params, target_config_layers):
    """
    Master Controller.
    Compares the loaded parameter shapes against the target configuration
    and applies Widening or Deepening as needed.
    
    Args:
        current_params: Dictionary of trained weights (from checkpoint).
                        Example keys: ['Dense_0', 'Dense_1', 'Dense_2']
        target_config_layers: List from config.
                              Example: [9, 128, 128, 128, 6]
                              (Input=9, H1=128, H2=128, H3=128, Output=6)
    """
    # --- A. Analyze Current Loaded Architecture ---
    # Flax layers are usually named 'Dense_0', 'Dense_1', etc.
    # We sort to ensure logical order (Dense_0 -> Dense_1 ...)
    layer_names = sorted(list(current_params.keys())) 
    
    # Identify Hidden vs Output
    # Assumption: The last Dense layer is ALWAYS the Output Layer.
    hidden_layers = layer_names[:-1] 
    output_layer = layer_names[-1]
    
    # Get current widths from Bias shapes
    # (Bias vector length == Neurons in that layer)
    current_widths = [current_params[l]['bias'].shape[0] for l in hidden_layers]
    
    # --- B. Analyze Target Architecture (CRITICAL FIX) ---
    # The config list is [Input, Hidden1, Hidden2, ..., Output].
    # We ONLY want to compare Hidden Layers.
    # Slice [1:-1] drops the Input (idx 0) and Output (idx -1).
    target_hidden_widths = list(target_config_layers)[1:-1]
    
    # Debug info (Uncomment if needed)
    # print(f"   [Net2Net Debug] Current Hidden: {current_widths}")
    # print(f"   [Net2Net Debug] Target Hidden:  {target_hidden_widths}")
    
    # --- C. CHECK FOR WIDENING ---
    needs_widening = False
    
    # Compare layer by layer (up to the minimum depth)
    limit = min(len(current_widths), len(target_hidden_widths))
    
    for i in range(limit):
        if current_widths[i] < target_hidden_widths[i]:
            needs_widening = True
            break
            
    if needs_widening:
        print(f"⚡ Net2Net Triggered: Widening Detected.")
        params = current_params
        
        # Iterate through all hidden layers and widen them if needed
        for i, layer in enumerate(hidden_layers):
            if i >= len(target_hidden_widths): break # Safety check
            
            # Determine the 'next' layer to adjust its input matrix
            # If we are at the last hidden layer, the 'next' is the Output Layer
            is_last_hidden = (i == len(hidden_layers) - 1)
            next_layer = output_layer if is_last_hidden else hidden_layers[i+1]
            
            target_w = target_hidden_widths[i]
            
            # Apply transformation
            params = widen_layer(params, layer, target_w, next_layer)
            
        current_params = params

    # --- D. CHECK FOR DEEPENING ---
    # If loaded model has fewer hidden layers than target config
    if len(current_widths) < len(target_hidden_widths):
        print(f"⚡ Net2Net Triggered: Deepening from {len(current_widths)} hidden layers to {len(target_hidden_widths)}")
        
        params = unfreeze(current_params)
        
        # Strategy: Insert new layers just before the Output Layer.
        
        # 1. Identify where the Output Layer currently lives
        old_out_name = output_layer
        
        # 2. Identify where the Output Layer SHOULD live
        # If target has 3 hidden layers, they occupy indices 0, 1, 2.
        # So the Output Layer should be at index 3.
        # Index = count of hidden layers.
        new_last_index = len(target_hidden_widths)
        new_out_name = f"Dense_{new_last_index}"
        
        print(f"      -> Moving Output Layer: {old_out_name} -> {new_out_name}")
        
        # Move the output layer weights to the new key
        params[new_out_name] = params.pop(old_out_name)
        
        # 3. Fill the gap with Identity Layers
        # We start filling from the first missing index
        start_idx = len(hidden_layers) 
        end_idx = new_last_index
        
        # Assume uniform width for new layers (take the first hidden width target)
        width = target_hidden_widths[0] 
        
        for i in range(start_idx, end_idx):
            new_layer = f"Dense_{i}"
            # Freeze/Unfreeze handling inside the loop
            params = deepen_network(freeze(params), new_layer, width)
            params = unfreeze(params)

        current_params = freeze(params)
        
        # Update our pointer to the output layer since we moved it
        output_layer = new_out_name

    # --- E. FINAL ALIGNMENT CHECK (The Safety Fix) ---
    # Ensure the final output layer accepts inputs matching the last hidden layer's width.
    # This catches edge cases where the widening/deepening logic might have missed the final link.
    if len(target_hidden_widths) > 0:
        last_hidden_width = target_hidden_widths[-1]
        current_params = align_output_input(current_params, output_layer, last_hidden_width)

    return current_params