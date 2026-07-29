"""Runnable demo: ``python -m algo_stack.preprocessing.standard_scaler.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.preprocessing.standard_scaler import StandardScaler


def main() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(loc=5.0, scale=3.0, size=(100, 3))
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)
    print("mean_ =", np.round(sc.mean_, 3))
    print("scale_ =", np.round(sc.scale_, 3))
    print("transformed mean ≈", np.round(Z.mean(axis=0), 8))
    print("transformed std  ≈", np.round(Z.std(axis=0), 8))
    X_hat = sc.inverse_transform(Z)
    print("round-trip max err =", float(np.max(np.abs(X - X_hat))))


if __name__ == "__main__":
    main()
