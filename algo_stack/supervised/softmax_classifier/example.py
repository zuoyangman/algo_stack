"""Runnable demo: ``python -m algo_stack.supervised.softmax_classifier.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.supervised.softmax_classifier import SoftmaxClassifier
from algo_stack.utils.preprocessing import StandardScaler, train_test_split


def main() -> None:
    rng = np.random.default_rng(0)
    centres = np.array([[-3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
    X = np.vstack([c + rng.normal(scale=0.5, size=(150, 2)) for c in centres])
    y = np.repeat(np.arange(3), 150)
    X = StandardScaler().fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

    for opt in ("sgd", "momentum", "adam"):
        clf = SoftmaxClassifier(
            optimizer=opt, learning_rate=0.1, n_epochs=300, random_state=0
        ).fit(Xtr, ytr)
        print(
            f"optimizer={opt:8s}  train_acc={clf.score(Xtr, ytr):.3f}  "
            f"test_acc={clf.score(Xte, yte):.3f}  final_loss={clf.loss_curve_[-1]:.4f}"
        )


if __name__ == "__main__":
    main()
