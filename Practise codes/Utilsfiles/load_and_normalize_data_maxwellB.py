# load_and_normalize_data_maxwellB.py

import os
import numpy as np
import torch
import jax.numpy as jnp
from sklearn.model_selection import train_test_split

def load_and_normalize_data_maxwellB(
    X_path, Y_path, seed=42, test_size=0.1, val_size=0.1
):
    """
    Custom data loader for Maxwell-B model.
    Uses per-component normalization for Y (stress) to avoid scale imbalance.
    Returns normalized X and Y, plus means/stds for de-normalization.
    """
    # Load tensors from .pt
    X = torch.load(X_path).numpy()
    Y = torch.load(Y_path).numpy()

    # Train/val/test split
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=seed
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_val, Y_train_val, test_size=val_size, random_state=seed
    )

    # Normalize X (per feature)
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std == 0] = 1  # avoid division by zero
    X_train_norm = (X_train - X_mean) / X_std
    X_val_norm   = (X_val   - X_mean) / X_std
    X_test_norm  = (X_test  - X_mean) / X_std

    # Normalize Y (per output dimension!)
    Y_mean = Y_train.mean(axis=0)    # shape (6,)
    Y_std  = Y_train.std(axis=0)
    Y_std[Y_std == 0] = 1
    Y_train_norm = (Y_train - Y_mean) / Y_std
    Y_val_norm   = (Y_val   - Y_mean) / Y_std
    Y_test_norm  = (Y_test  - Y_mean) / Y_std

    return (
        jnp.array(X_train_norm), jnp.array(X_val_norm), jnp.array(X_test_norm),
        jnp.array(Y_train_norm), jnp.array(Y_val_norm), jnp.array(Y_test_norm),
        X_mean, X_std, Y_mean, Y_std
    )