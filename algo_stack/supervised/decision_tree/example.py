"""Runnable demo: ``python -m algo_stack.supervised.decision_tree.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.supervised.decision_tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)
from algo_stack.utils.preprocessing import train_test_split


def main() -> None:
    rng = np.random.default_rng(0)

    print("--- Decision tree: XOR ---")
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])
    for depth in (1, 2):
        clf = DecisionTreeClassifier(max_depth=depth).fit(X, y)
        print(f"max_depth={depth}  acc={clf.score(X, y):.3f}  pred={clf.predict(X)}")

    print("\n--- Decision tree: 2 blobs ---")
    X = np.vstack(
        [
            rng.normal([-2, 0], 0.5, size=(80, 2)),
            rng.normal([2, 0], 0.5, size=(80, 2)),
        ]
    )
    y = np.repeat([0, 1], 80)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    clf = DecisionTreeClassifier(max_depth=4).fit(Xtr, ytr)
    print(f"train_acc={clf.score(Xtr, ytr):.3f}  test_acc={clf.score(Xte, yte):.3f}")

    print("\n--- Decision tree regression: noisy sine ---")
    X = rng.uniform(-3, 3, size=(200, 1))
    y = np.sin(X.ravel()) + 0.1 * rng.normal(size=200)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    reg = DecisionTreeRegressor(max_depth=6, min_samples_leaf=3).fit(Xtr, ytr)
    print(f"train_R²={reg.score(Xtr, ytr):.3f}  test_R²={reg.score(Xte, yte):.3f}")


if __name__ == "__main__":
    main()
