"""Runnable demo: ``python -m algo_stack.unsupervised.kde.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.unsupervised.kde import KernelDensity


def main() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 1))
    kde = KernelDensity(bandwidth=0.4).fit(X)
    grid = np.linspace(-3, 3, 7).reshape(-1, 1)
    print("log density =", np.round(kde.score_samples(grid), 3))
    print("mean loglik =", round(kde.score(X), 3))


if __name__ == "__main__":
    main()
