"""Runnable demo: ``python -m algo_stack.supervised.perceptron.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.supervised.perceptron import Perceptron


def main() -> None:
    rng = np.random.default_rng(0)

    print("--- Binary Perceptron (linearly separable) ---")
    X = np.vstack([rng.normal([-2, 0], 0.3, size=(60, 2)),
                   rng.normal([+2, 0], 0.3, size=(60, 2))])
    y = np.array([0] * 60 + [1] * 60)
    clf = Perceptron(learning_rate=1.0, n_iter=50, random_state=0).fit(X, y)
    print(f"coef_={np.round(clf.coef_, 3)}, intercept_={clf.intercept_:.3f}")
    print(f"accuracy={clf.score(X, y):.3f}, epochs={clf.n_iter_}")
    print(f"mistakes last epoch={clf.mistakes_history_[-1]}")

    print("\n--- Multiclass Perceptron (3 blobs, OvR) ---")
    centres = np.array([[-3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
    X = np.vstack([c + rng.normal(scale=0.4, size=(50, 2)) for c in centres])
    y = np.repeat(np.arange(3), 50)
    clf = Perceptron(learning_rate=1.0, n_iter=100, random_state=0).fit(X, y)
    print(f"coef_.shape={clf.coef_.shape}")
    print(f"accuracy={clf.score(X, y):.3f}, epochs={clf.n_iter_}")


if __name__ == "__main__":
    main()
