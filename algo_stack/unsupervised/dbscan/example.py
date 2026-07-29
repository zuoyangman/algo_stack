"""Runnable demo: ``python -m algo_stack.unsupervised.dbscan.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.unsupervised.dbscan import DBSCAN


def main() -> None:
    rng = np.random.default_rng(0)
    A = rng.normal(size=(80, 2)) + np.array([-3.0, 0.0])
    B = rng.normal(size=(80, 2)) + np.array([3.0, 0.0])
    noise = rng.uniform(-6, 6, size=(20, 2))
    X = np.vstack([A, B, noise])
    db = DBSCAN(eps=0.8, min_samples=5).fit(X)
    labels = db.labels_
    n_clusters = len(set(labels) - {-1})
    n_noise = int(np.sum(labels == -1))
    print("clusters =", n_clusters, "noise =", n_noise)
    print("n_core =", len(db.core_sample_indices_))


if __name__ == "__main__":
    main()
