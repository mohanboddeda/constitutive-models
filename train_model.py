import os
import numpy as np
import torch
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import hydra
from omegaconf import DictConfig
from functools import partial

# Simple feedforward neural network with Flax
class MLP(nn.Module):
    features: list  # e.g. [128, 128, output_dim]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
        x = nn.Dense(self.features[-1])(x)  # final output layer
        return x

# Mean squared error loss
@jax.jit
def mse_loss(params, model, x, y):
    preds = model.apply(params, x)
    return jnp.mean((preds - y) ** 2)

# Single training step
@jax.jit
def train_step(params, model, opt_state, x, y, optimizer):
    loss, grads = jax.value_and_grad(mse_loss)(params, model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@hydra.main(config_path="config/train", config_name="trainConfig", version_base=None)
def main(cfg: DictConfig):
    print(f"Loading data from {cfg.data_path_X} and {cfg.data_path_Y} ...")
    X = torch.load(cfg.data_path_X).numpy()
    Y = torch.load(cfg.data_path_Y).numpy()

    # Convert to jax arrays
    X = jnp.array(X, dtype=jnp.float32)
    Y = jnp.array(Y, dtype=jnp.float32)

    print(f"Data shapes: X {X.shape}, Y {Y.shape}")

    # Create model and initialize parameters
    key = jax.random.PRNGKey(cfg.seed)
    model = MLP(features=cfg.model.layers)
    params = model.init(key, jnp.ones([1, X.shape[1]]))

    # Create optimizer
    optimizer = optax.adam(cfg.training.learning_rate)
    opt_state = optimizer.init(params)

    # Create batching function
    batch_size = cfg.training.batch_size
    num_batches = int(np.ceil(X.shape[0] / batch_size))

    @partial(jax.jit, static_argnums=3)
    def batch_train(params, opt_state, x_batch, y_batch):
        return train_step(params, model, opt_state, x_batch, y_batch, optimizer)

    for epoch in range(cfg.training.num_epochs):
        # Shuffle data each epoch
        perm = np.random.permutation(X.shape[0])
        X_shuffled = X[perm]
        Y_shuffled = Y[perm]

        epoch_loss = 0
        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, X.shape[0])
            x_batch = X_shuffled[start:end]
            y_batch = Y_shuffled[start:end]

            params, opt_state, loss = batch_train(params, opt_state, x_batch, y_batch)
            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{cfg.training.num_epochs} - loss: {avg_loss:.6f}")

    # Optionally save trained parameters
    params_path = os.path.join(cfg.output_dir, "trained_params.npz")
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    np.savez(params_path, **jax.tree_util.tree_map(lambda x: np.array(x), params))
    print(f"Trained parameters saved to {params_path}")

if __name__ == "__main__":
    main()