"""Runnable demo: ``python -m algo_stack.unsupervised.gmm.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.unsupervised.gmm import GaussianMixture


def main() -> None:
    rng = np.random.default_rng(0)
    X = np.vstack(
        [
            rng.normal([-2, 0], 0.4, size=(100, 2)),
            rng.normal([2, 0], 0.4, size=(100, 2)),
        ]
    )
    gmm = GaussianMixture(
        n_components=2, covariance_type="diag", random_state=0
    ).fit(X)
    print("weights =", np.round(gmm.weights_, 3))
    print("means   =")
    print(np.round(gmm.means_[np.lexsort(gmm.means_.T)], 3))
    print("avg loglik =", round(gmm.score(X), 3))


if __name__ == "__main__":
    main()
