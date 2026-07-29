"""Runnable demo: ``python -m algo_stack.unsupervised.denoising_autoencoder.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.unsupervised.denoising_autoencoder import DenoisingAutoencoder


def main() -> None:
    rng = np.random.default_rng(0)
    n_samples, n_features = 500, 20
    latent = rng.normal(size=(n_samples, 4))
    W = rng.normal(size=(4, n_features))
    X = latent @ W + 0.05 * rng.normal(size=(n_samples, n_features))

    dae = DenoisingAutoencoder(
        encoding_dim=4,
        hidden_dim=16,
        noise_std=0.2,
        learning_rate=0.01,
        n_epochs=150,
        batch_size=32,
        random_state=0,
    ).fit(X)

    Z = dae.transform(X)
    X_hat = dae.reconstruct(X)
    mse = float(np.mean((X - X_hat) ** 2))
    print(f"encoding_dim={dae.encoding_dim}  bottleneck shape={Z.shape}")
    print(f"noise_std={dae.noise_std}  reconstruction MSE={mse:.6f}")
    print(f"final_loss={dae.loss_curve_[-1]:.6f}  epochs={dae.n_iter_}")


if __name__ == "__main__":
    main()
