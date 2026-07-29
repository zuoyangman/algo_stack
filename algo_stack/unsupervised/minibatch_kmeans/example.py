"""Runnable demo: ``python -m algo_stack.unsupervised.minibatch_kmeans.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.unsupervised.minibatch_kmeans import MiniBatchKMeans


def main() -> None:
    rng = np.random.default_rng(0)
    centres_true = np.array([[-4.0, -4.0], [0.0, 4.0], [4.0, -4.0]])
    X = np.vstack(
        [c + rng.normal(scale=0.6, size=(200, 2)) for c in centres_true]
    )
    mb = MiniBatchKMeans(
        n_clusters=3, batch_size=50, random_state=0, n_init=5, max_iter=50
    ).fit(X)
    print("Inertia =", round(mb.inertia_, 3))
    print("Centres =")
    print(np.round(mb.cluster_centers_[np.lexsort(mb.cluster_centers_.T)], 3))


if __name__ == "__main__":
    main()
