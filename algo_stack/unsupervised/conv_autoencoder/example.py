"""Runnable demo: ``python -m algo_stack.unsupervised.conv_autoencoder.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.unsupervised.conv_autoencoder import ConvAutoencoder


def main() -> None:
    rng = np.random.default_rng(0)
    n, size = 60, 8
    # Synthetic bars: horizontal vs vertical structure in latent content.
    X = []
    for i in range(n):
        img = rng.normal(0, 0.05, size=(size, size))
        if i < n // 2:
            img[size // 2, :] += 1.0
        else:
            img[:, size // 2] += 1.0
        X.append(img)
    X = np.asarray(X)[:, None, :, :]

    cae = ConvAutoencoder(
        encoding_dim=4,
        num_filters=4,
        learning_rate=0.08,
        n_epochs=60,
        batch_size=16,
        random_state=0,
    ).fit(X)
    Z = cae.transform(X)
    X_hat = cae.reconstruct(X)
    mse = float(np.mean((X - X_hat) ** 2))
    print(f"bottleneck shape={Z.shape}")
    print(f"reconstruction MSE={mse:.6f}")
    print(f"final_loss={cae.loss_curve_[-1]:.6f}  epochs={cae.n_iter_}")


if __name__ == "__main__":
    main()
