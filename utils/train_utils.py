import os
import numpy as np
import torch
import jax
import jax.numpy as jnp
import flax
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score, mean_absolute_error

def load_and_normalize_data(X_path, Y_path, seed=42, test_size=0.1, val_size=0.1):
    X = torch.load(X_path).numpy()
    Y = torch.load(Y_path).numpy()
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=val_size, random_state=seed)

    # Normalize X
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std == 0] = 1
    X_train = (X_train - X_mean) / X_std
    X_val   = (X_val - X_mean) / X_std
    X_test  = (X_test - X_mean) / X_std

    # Normalize Y
    Y_mean = Y_train.mean(axis=0)
    Y_std = Y_train.std(axis=0)
    Y_std[Y_std == 0] = 1
    Y_train = (Y_train - Y_mean) / Y_std
    Y_val   = (Y_val - Y_mean) / Y_std
    Y_test  = (Y_test - Y_mean) / Y_std

    
    return (jnp.array(X_train), jnp.array(X_val), jnp.array(X_test),
            jnp.array(Y_train), jnp.array(Y_val), jnp.array(Y_test),
            X_mean, X_std, Y_mean, Y_std)

def save_checkpoint(params, X_mean, X_std, Y_mean, Y_std, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    to_save = {
        "params": params,
        "X_mean": X_mean,
        "X_std": X_std,
        "Y_mean": Y_mean,
        "Y_std": Y_std
    }
    with open(path, "wb") as f:
        f.write(flax.serialization.to_bytes(to_save))

def load_checkpoint(path, init_params):
    with open(path, "rb") as f:
        restored = flax.serialization.from_bytes(init_params, f.read())
    return restored

def plot_learning_curves(train_losses, val_losses, fig_dir, model_type):
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,  label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Learning Curves ({model_type})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "learning_curves.png"))
    plt.close()

def plot_residual_hist(residuals, fig_dir, model_type):
    residuals_1d = np.ravel(residuals)
    plt.figure(figsize=(8,5))
    sns.histplot(residuals_1d, bins=30, color='skyblue', stat='density')
    kde = gaussian_kde(residuals_1d)
    x_range = np.linspace(residuals_1d.min(), residuals_1d.max(), 1000)
    plt.plot(x_range, kde(x_range), color='orange', lw=2, label='KDE')
    plt.axvline(0, color='red', linestyle='--', label='Zero Residual')
    plt.title(f"Residuals on Test Data ({model_type})")
    plt.xlabel("Residual (True - Predicted)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "residual_histogram.png"))
    plt.close()

def plot_residuals_vs_pred(y_pred, residuals, fig_dir, model_type):
    y_pred_1d = np.ravel(y_pred)
    residuals_1d = np.ravel(residuals)
    plt.figure(figsize=(8,5))
    plt.scatter(y_pred_1d, residuals_1d, alpha=0.5)
    smoothed = sm.nonparametric.lowess(residuals_1d, y_pred_1d, frac=0.3)
    plt.plot(smoothed[:,0], smoothed[:,1], color='red', lw=2, label='LOWESS')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residual (True - Predicted)")
    plt.title(f"Residuals vs Predicted ({model_type})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "residuals_vs_predictions.png"))
    plt.close()