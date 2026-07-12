"""Runnable demo: ``python -m algo_stack.supervised.rnn.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.supervised.rnn import RNNClassifier
from algo_stack.utils.preprocessing import train_test_split


def make_sequence_data(
    n_per_class: int = 100, seq_len: int = 10, n_features: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Class 0: rising mean over time; class 1: falling mean."""

    rng = np.random.default_rng(0)
    X0, X1 = [], []
    for _ in range(n_per_class):
        trend = np.linspace(-1, 1, seq_len)[:, None]
        X0.append(rng.normal(size=(seq_len, n_features)) + trend)
        X1.append(rng.normal(size=(seq_len, n_features)) - trend)
    X = np.array(X0 + X1)
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y


def main() -> None:
    X, y = make_sequence_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

    clf = RNNClassifier(
        hidden_size=16, learning_rate=0.05, n_epochs=80, batch_size=32, random_state=0
    ).fit(Xtr, ytr)
    print(f"train_acc={clf.score(Xtr, ytr):.3f}  test_acc={clf.score(Xte, yte):.3f}")
    print(f"final_loss={clf.loss_curve_[-1]:.4f}  epochs={clf.n_iter_}")
    print(f"W_xh_.shape={clf.W_xh_.shape}  W_hh_.shape={clf.W_hh_.shape}")


if __name__ == "__main__":
    main()
