"""Runnable demo: ``python -m algo_stack.unsupervised.pca.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.unsupervised.pca import PCA


def main() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    X[:, 1] = 2.0 * X[:, 0] + 0.1 * rng.normal(size=200)
    pca = PCA(n_components=2).fit(X)
    print("explained_variance_ratio_ =", np.round(pca.explained_variance_ratio_, 3))
    Z = pca.transform(X)
    X_hat = pca.inverse_transform(Z)
    mse = float(np.mean((X - X_hat) ** 2))
    print("recon MSE (2 comps) =", round(mse, 4))


if __name__ == "__main__":
    main()
