"""Runnable demo: ``python -m algo_stack.unsupervised.umap.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.unsupervised.umap import UMAP


def main() -> None:
    rng = np.random.default_rng(0)
    X = np.vstack(
        [
            rng.normal([-3, 0, 0], 0.35, size=(40, 3)),
            rng.normal([3, 0, 0], 0.35, size=(40, 3)),
        ]
    )
    umap = UMAP(
        n_components=2,
        n_neighbors=10,
        min_dist=0.1,
        n_epochs=120,
        learning_rate=1.0,
        random_state=0,
    )
    Y = umap.fit_transform(X)
    c0, c1 = Y[:40].mean(axis=0), Y[40:].mean(axis=0)
    print("embedding shape =", Y.shape)
    print("centroid distance =", round(float(np.linalg.norm(c0 - c1)), 4))
    print("a, b =", round(umap.a_, 4), round(umap.b_, 4))


if __name__ == "__main__":
    main()
