import os
import numpy as np
import torch
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import hydra
import flax.serialization
import matplotlib.pyplot as plt
import seaborn as sns  # seaborn for easy KDE plotting
import statsmodels.api as sm  # for LOWESS smoothing

from omegaconf import DictConfig
from functools import partial
from sklearn.metrics import r2_score, mean_absolute_error

# For splitting dataset into train/val/test
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde
 
# =============================================================================
# 1. Define your MLP (Multi-Layer Perceptron) model with Flax
# =============================================================================

class MLP(nn.Module):
    features: list  # List of layer sizes, e.g. [128, 128, output_dim]

    @nn.compact
    def __call__(self, x):
        # Pass through all hidden layers with ReLU activations
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
        # Final output layer - linear
        x = nn.Dense(self.features[-1])(x)
        return x

# =============================================================================
# 2. Loss Function: Mean squared error loss, jit-compiled for speed
# =============================================================================

# =============================================================================
# 3. One training step: forward, loss, backward, update params
# =============================================================================

def make_train_step(model, optimizer):
    @jax.jit
    def mse_loss(params, x, y):
        preds = model.apply(params, x)
        return jnp.mean((preds - y) ** 2)

    @jax.jit
    def train_step(params, opt_state, x, y):
        loss = mse_loss(params, x, y)
        grads = jax.grad(mse_loss)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    
     # Predictor function you can pass around for evaluation
    def predict_fn(params, x):
        return model.apply(params, x)

    return train_step, mse_loss, predict_fn

# Compute average loss over a dataset (val or test), batched
def compute_loss(params, predict_fn, X_data, Y_data, batch_size):
    num_batches = int(np.ceil(X_data.shape[0] / batch_size))
    total_loss = 0.0
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, X_data.shape[0])
        x_batch = X_data[start:end]
        y_batch = Y_data[start:end]

        preds = predict_fn(params, x_batch)
        batch_loss = jnp.mean((preds - y_batch) ** 2).item()

        total_loss += batch_loss * (end - start)
    return total_loss / X_data.shape[0]

# =============================================================================
# 4. Main Function
# =============================================================================

@hydra.main(config_path="config/train", config_name="trainConfig", version_base=None)
def main(cfg: DictConfig):

    model_data = {
        "carreau_yasuda": {
            "X": "./datafiles/X_3D_carreau_yasuda.pt",
            "Y": "./datafiles/Y_3D_carreau_yasuda.pt",
            "output_dir": "./trained_models/carreau_yasuda",
            "output_dim": 1
        },
        "maxwell_B": {
            "X": "./datafiles/X_3D_maxwell_B.pt",
            "Y": "./datafiles/Y_3D_maxwell_B.pt",
            "output_dir": "./trained_models/maxwell_B",
            "output_dim": 6   # symmetric tensor components
        },
        "oldroyd_B": {
            "X": "./datafiles/X_3D_oldroyd_B.pt",
            "Y": "./datafiles/Y_3D_oldroyd_B.pt",
            "output_dir": "./trained_models/oldroyd_B",
            "output_dim": 6
        }
    }
    
    if cfg.model_type not in model_data:
        raise ValueError(f"Unsupported model_type: {cfg.model_type}")

    data_paths = model_data[cfg.model_type]
    print(f"Training model: {cfg.model_type}")
    print(f"Loading data from {data_paths['X']} and {data_paths['Y']} ...")

    # --- Loading and preparing data ---
    X = torch.load(data_paths["X"]).numpy()
    Y = torch.load(data_paths["Y"]).numpy()
    X = jnp.array(X, dtype=jnp.float32)
    Y = jnp.array(Y, dtype=jnp.float32)

    print(f"Data shapes: X {X.shape}, Y {Y.shape}")

    # Set output directory dynamically
    cfg.output_dir = data_paths["output_dir"]
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Split dataset
    X_interim, X_test, Y_interim, Y_test = train_test_split(
        np.array(X), np.array(Y), test_size=0.1, random_state=cfg.seed)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_interim, Y_interim, test_size=0.1, random_state=cfg.seed)

    X_train = jnp.array(X_train)
    Y_train = jnp.array(Y_train)
    X_val = jnp.array(X_val)
    Y_val = jnp.array(Y_val)
    X_test = jnp.array(X_test)
    Y_test = jnp.array(Y_test)

    # Adjust output layer size based on targets Y shape or config
    output_dim = model_data[cfg.model_type]["output_dim"]
    model_layers = list(cfg.model.layers)
    if model_layers[-1] != output_dim:
        print(f"Adjusting output layer from {model_layers[-1]} to {output_dim}")
        model_layers[-1] = output_dim

    key = jax.random.PRNGKey(cfg.seed)
    model = MLP(features=model_layers)
    params = model.init(key, jnp.ones([1, X.shape[1]]))

    optimizer = optax.adam(cfg.training.learning_rate)
    opt_state = optimizer.init(params)

    train_step, mse_loss, predict_fn = make_train_step(model, optimizer)

    batch_size = cfg.training.batch_size
    num_batches = int(np.ceil(X_train.shape[0] / batch_size))

    @jax.jit
    def batch_train(params, opt_state, x, y):
        return train_step(params, opt_state, x, y)

    best_val_loss = float('inf')
    patience = cfg.training.get('patience', 10)
    patience_counter = 0

    train_losses = []
    val_losses = []

    fig_dir = os.path.join(cfg.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # --- Training loop ---
    for epoch in range(cfg.training.num_epochs):
        perm = np.random.permutation(X_train.shape[0])
        X_shuffled = X_train[perm]
        Y_shuffled = Y_train[perm]

        epoch_loss = 0.0
        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, X_train.shape[0])
            x_batch = X_shuffled[start:end]
            y_batch = Y_shuffled[start:end]

            params, opt_state, loss = batch_train(params, opt_state, x_batch, y_batch)
            epoch_loss += loss.item() * (end - start)

        avg_train_loss = epoch_loss / X_train.shape[0]
        avg_val_loss = compute_loss(params, predict_fn, X_val, Y_val, batch_size)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{cfg.training.num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # Early stopping and checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            params_path = os.path.join(cfg.output_dir, "trained_params.msgpack")
            with open(params_path, "wb") as f:
                f.write(flax.serialization.to_bytes(params))
            print(f"Checkpoint saved at epoch {epoch + 1}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # --- Plot learning curves ---
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'Learning Curves for {cfg.model_type}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "learning_curves.png"))
    plt.close()

    # --- Load best model for test evaluation ---
    print("Loading best model parameters for test evaluation...")
    init_params = model.init(key, jnp.ones([1, X.shape[1]]))
    with open(params_path, "rb") as f:
        params_bytes = f.read()
    best_params = flax.serialization.from_bytes(init_params, params_bytes)

    def predict(params, model, X):
        preds = model.apply(params, X)
        return np.array(preds)

    y_pred = predict(best_params, model, X_test)
    y_true = np.array(Y_test)

    residuals = y_true - y_pred

    # --- Plot residual histogram ---
    residuals_1d = np.ravel(residuals)  # flatten

    plt.figure(figsize=(8, 5))
    sns.histplot(residuals_1d, bins=30, color='skyblue', stat='density')

    kde = gaussian_kde(residuals_1d)
    x_range = np.linspace(residuals_1d.min(), residuals_1d.max(), 1000)
    kde_vals = kde(x_range)
    plt.plot(x_range, kde_vals, color='orange', linewidth=2, label='KDE')

    plt.axvline(0, color='red', linestyle='--', label='Zero Residual')
    plt.title(f'Histogram and Density of Residuals on Test Data ({cfg.model_type})')
    plt.xlabel('Residual (True - Predicted)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "residual_histogram.png"))
    plt.close()

    # --- Plot residuals vs predictions ---
    y_pred_1d = np.ravel(y_pred)

    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred_1d, residuals_1d, alpha=0.5, label='Residuals')

    smoothed = sm.nonparametric.lowess(residuals_1d, y_pred_1d, frac=0.3)

    plt.plot(smoothed[:, 0], smoothed[:, 1], color='red', linewidth=2, label='LOWESS Smoother')

    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals (True - Predicted)")
    plt.title(f"Residuals vs Predicted Values with Smooth Trend ({cfg.model_type})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "residuals_vs_predictions_enhanced.png"))
    plt.close()

    # --- Compute test metrics ---
    test_loss = compute_loss(best_params, predict_fn, X_test, Y_test, batch_size)
    print(f"Test Loss: {test_loss:.6f}")

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"Test R²: {r2:.4f}")
    print(f"Test MAE: {mae:.6f}")

if __name__ == "__main__":
    main()