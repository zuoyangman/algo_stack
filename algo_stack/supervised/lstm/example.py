"""Runnable demo: ``python -m algo_stack.supervised.lstm.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.supervised.lstm import GRUClassifier, LSTMClassifier
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

    lstm = LSTMClassifier(
        hidden_size=16, learning_rate=0.05, n_epochs=60, batch_size=32, random_state=0
    ).fit(Xtr, ytr)
    print(
        f"LSTM  train_acc={lstm.score(Xtr, ytr):.3f}  "
        f"test_acc={lstm.score(Xte, yte):.3f}  "
        f"final_loss={lstm.loss_curve_[-1]:.4f}"
    )

    gru = GRUClassifier(
        hidden_size=16, learning_rate=0.05, n_epochs=60, batch_size=32, random_state=0
    ).fit(Xtr, ytr)
    print(
        f"GRU   train_acc={gru.score(Xtr, ytr):.3f}  "
        f"test_acc={gru.score(Xte, yte):.3f}  "
        f"final_loss={gru.loss_curve_[-1]:.4f}"
    )


if __name__ == "__main__":
    main()
