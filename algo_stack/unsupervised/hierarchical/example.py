"""Runnable demo: ``python -m algo_stack.unsupervised.hierarchical.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.unsupervised.hierarchical import AgglomerativeClustering


def main() -> None:
    rng = np.random.default_rng(0)
    X = np.vstack(
        [
            rng.normal([-3, 0], 0.3, size=(40, 2)),
            rng.normal([3, 0], 0.3, size=(40, 2)),
            rng.normal([0, 3], 0.3, size=(40, 2)),
        ]
    )
    for linkage in ("ward", "average", "complete", "single"):
        model = AgglomerativeClustering(n_clusters=3, linkage=linkage).fit(X)
        print(linkage, "labels unique =", sorted(set(model.labels_)))


if __name__ == "__main__":
    main()
