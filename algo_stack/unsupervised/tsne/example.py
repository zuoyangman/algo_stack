"""Runnable demo: ``python -m algo_stack.unsupervised.tsne.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.unsupervised.tsne import TSNE


def main() -> None:
    rng = np.random.default_rng(0)
    X = np.vstack(
        [
            rng.normal([-3, 0, 0], 0.4, size=(40, 3)),
            rng.normal([3, 0, 0], 0.4, size=(40, 3)),
        ]
    )
    tsne = TSNE(
        n_components=2, perplexity=15, n_iter=400, random_state=0, learning_rate=100
    )
    Y = tsne.fit_transform(X)
    print("embedding shape =", Y.shape)
    print("KL =", round(tsne.kl_divergence_, 4))


if __name__ == "__main__":
    main()
