"""Runnable demo: ``python -m algo_stack.supervised.gradient_boosting.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.supervised.gradient_boosting import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from algo_stack.utils.preprocessing import train_test_split


def main() -> None:
    rng = np.random.default_rng(0)

    print("--- GB regression: noisy sine ---")
    X = rng.uniform(-np.pi, np.pi, size=(300, 1))
    y = np.sin(X.ravel()) + 0.15 * rng.normal(size=300)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    reg = GradientBoostingRegressor(
        n_estimators=80, learning_rate=0.1, max_depth=3, random_state=0
    ).fit(Xtr, ytr)
    print(f"train_R²={reg.score(Xtr, ytr):.3f}  test_R²={reg.score(Xte, yte):.3f}")

    print("\n--- GB classification: blobs ---")
    X = np.vstack(
        [
            rng.normal([-2, 0], 0.7, size=(100, 2)),
            rng.normal([2, 0], 0.7, size=(100, 2)),
        ]
    )
    y = np.repeat([0, 1], 100)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    clf = GradientBoostingClassifier(
        n_estimators=40, learning_rate=0.1, max_depth=2, random_state=0
    ).fit(Xtr, ytr)
    print(f"train_acc={clf.score(Xtr, ytr):.3f}  test_acc={clf.score(Xte, yte):.3f}")


if __name__ == "__main__":
    main()
