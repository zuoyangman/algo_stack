"""Runnable demo: ``python -m algo_stack.supervised.knn.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.supervised.knn import KNNClassifier, KNNRegressor
from algo_stack.utils.preprocessing import train_test_split


def main() -> None:
    rng = np.random.default_rng(0)

    print("--- KNN classification (3 well-separated blobs) ---")
    centres = np.array([[-3, 0.0], [0.0, 3.0], [3.0, 0.0]])
    X = np.vstack([centres[k] + rng.normal(scale=0.6, size=(80, 2)) for k in range(3)])
    y = np.repeat(np.arange(3), 80)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    for k in (1, 5, 15):
        clf = KNNClassifier(n_neighbors=k).fit(Xtr, ytr)
        print(f"k={k:2d}  train_acc={clf.score(Xtr, ytr):.3f}  test_acc={clf.score(Xte, yte):.3f}")

    print("\n--- KNN regression (noisy sine) ---")
    X = rng.uniform(-3, 3, size=(200, 1))
    y = np.sin(X.ravel()) + 0.1 * rng.normal(size=X.shape[0])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    for k in (1, 5, 25):
        reg = KNNRegressor(n_neighbors=k, weights="distance").fit(Xtr, ytr)
        print(f"k={k:2d}  train_R²={reg.score(Xtr, ytr):.3f}  test_R²={reg.score(Xte, yte):.3f}")


if __name__ == "__main__":
    main()
