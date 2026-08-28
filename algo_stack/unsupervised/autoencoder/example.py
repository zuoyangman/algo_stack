"""Runnable demo: ``python -m algo_stack.unsupervised.autoencoder.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.unsupervised.autoencoder import Autoencoder


def main() -> None:
    rng = np.random.default_rng(0)
    n_samples, n_features = 500, 20
    # Low-rank data: 4 latent factors
    latent = rng.normal(size=(n_samples, 4))
    W = rng.normal(size=(4, n_features))
    X = latent @ W + 0.05 * rng.normal(size=(n_samples, n_features))

    ae = Autoencoder(
        encoding_dim=4,
        hidden_dim=16,
        learning_rate=0.01,
        n_epochs=150,
        batch_size=32,
        random_state=0,
    ).fit(X)

    Z = ae.transform(X)
    X_hat = ae.reconstruct(X)
    mse = float(np.mean((X - X_hat) ** 2))
    print(f"encoding_dim={ae.encoding_dim}  bottleneck shape={Z.shape}")
    print(f"reconstruction MSE={mse:.6f}")
    print(f"final_loss={ae.loss_curve_[-1]:.6f}  epochs={ae.n_iter_}")


if __name__ == "__main__":
    main()
