"""Runnable demo: ``python -m algo_stack.unsupervised.kmeans.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.unsupervised.kmeans import KMeans


def main() -> None:
    rng = np.random.default_rng(0)
    centres_true = np.array([[-4.0, -4.0], [0.0, 4.0], [4.0, -4.0]])
    X = np.vstack(
        [c + rng.normal(scale=0.6, size=(120, 2)) for c in centres_true]
    )

    km = KMeans(n_clusters=3, random_state=0, n_init=10).fit(X)
    print("Inertia          =", round(km.inertia_, 3))
    print("Iterations (best)=", km.n_iter_)
    print("Centres (sorted) =")
    order = np.lexsort(km.cluster_centers_.T)
    print(np.round(km.cluster_centers_[order], 3))
    print("True centres     =")
    print(np.round(centres_true[np.lexsort(centres_true.T)], 3))

    # Predict on a fresh point
    test = np.array([[-3.5, -3.8], [0.1, 4.2], [3.8, -4.1]])
    print("Predicted labels for test points:", km.predict(test))


if __name__ == "__main__":
    main()
