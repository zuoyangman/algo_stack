"""Runnable demo: ``python -m algo_stack.supervised.random_forest.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.supervised.decision_tree import DecisionTreeClassifier
from algo_stack.supervised.random_forest import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from algo_stack.utils.preprocessing import train_test_split


def main() -> None:
    rng = np.random.default_rng(0)

    print("--- RF vs single tree on noisy blobs ---")
    X = np.vstack(
        [
            rng.normal([-2, 0], 1.0, size=(100, 2)),
            rng.normal([2, 0], 1.0, size=(100, 2)),
        ]
    )
    y = np.repeat([0, 1], 100)
    flip = rng.random(len(y)) < 0.15
    y = np.where(flip, 1 - y, y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

    tree = DecisionTreeClassifier(max_depth=None, random_state=0).fit(Xtr, ytr)
    rf = RandomForestClassifier(
        n_estimators=40, max_depth=None, random_state=0
    ).fit(Xtr, ytr)
    print(f"tree test_acc={tree.score(Xte, yte):.3f}")
    print(f"RF   test_acc={rf.score(Xte, yte):.3f}")

    print("\n--- RF regression: noisy sine ---")
    X = rng.uniform(-3, 3, size=(250, 1))
    y = np.sin(X.ravel()) + 0.2 * rng.normal(size=250)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1)
    reg = RandomForestRegressor(
        n_estimators=30, max_depth=6, max_features=1, random_state=0
    ).fit(Xtr, ytr)
    print(f"train_R²={reg.score(Xtr, ytr):.3f}  test_R²={reg.score(Xte, yte):.3f}")


if __name__ == "__main__":
    main()
