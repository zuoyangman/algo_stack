"""Runnable demo: ``python -m algo_stack.unsupervised.vae.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.unsupervised.vae import VariationalAutoencoder


def main() -> None:
    rng = np.random.default_rng(0)
    n_samples, n_features = 500, 20
    latent = rng.normal(size=(n_samples, 4))
    W = rng.normal(size=(4, n_features))
    X = latent @ W + 0.05 * rng.normal(size=(n_samples, n_features))

    vae = VariationalAutoencoder(
        encoding_dim=4,
        hidden_dim=16,
        learning_rate=0.005,
        n_epochs=150,
        batch_size=32,
        beta=1.0,
        random_state=0,
    ).fit(X)

    Z = vae.transform(X)
    X_hat = vae.reconstruct(X)
    samples = vae.sample(8)
    mse = float(np.mean((X - X_hat) ** 2))
    print(f"encoding_dim={vae.encoding_dim}  μ shape={Z.shape}")
    print(f"reconstruction MSE={mse:.6f}")
    print(f"sample shape={samples.shape}")
    print(f"final_loss={vae.loss_curve_[-1]:.6f}  epochs={vae.n_iter_}")


if __name__ == "__main__":
    main()
