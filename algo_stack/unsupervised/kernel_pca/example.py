"""Runnable demo: ``python -m algo_stack.unsupervised.kernel_pca.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.unsupervised.kernel_pca import KernelPCA


def main() -> None:
    rng = np.random.default_rng(0)
    t = rng.uniform(0, 2 * np.pi, size=150)
    X = np.column_stack([np.cos(t), np.sin(t)]) + 0.05 * rng.normal(size=(150, 2))
    kpca = KernelPCA(n_components=2, kernel="rbf", gamma=2.0).fit(X)
    Z = kpca.transform(X)
    print("embedding shape =", Z.shape)
    print("lambdas =", np.round(kpca.lambdas_, 3))


if __name__ == "__main__":
    main()
