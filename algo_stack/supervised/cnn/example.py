"""Runnable demo: ``python -m algo_stack.supervised.cnn.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.supervised.cnn import CNNClassifier
from algo_stack.utils.preprocessing import train_test_split


def make_pattern_data(n_per_class: int = 80, size: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Class 0 = horizontal bar; class 1 = vertical bar (8×8 grayscale)."""

    rng = np.random.default_rng(0)
    X0, X1 = [], []
    for _ in range(n_per_class):
        img0 = rng.normal(0, 0.1, size=(size, size))
        img0[size // 2, :] += 2.0
        X0.append(img0)
        img1 = rng.normal(0, 0.1, size=(size, size))
        img1[:, size // 2] += 2.0
        X1.append(img1)
    X = np.array(X0 + X1)[:, None, :, :]  # (N, 1, H, W)
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y


def main() -> None:
    X, y = make_pattern_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

    clf = CNNClassifier(
        num_filters=4, filter_size=3, pool_size=2,
        learning_rate=0.1, n_epochs=60, batch_size=16, random_state=0,
    ).fit(Xtr, ytr)
    print(f"train_acc={clf.score(Xtr, ytr):.3f}  test_acc={clf.score(Xte, yte):.3f}")
    print(f"final_loss={clf.loss_curve_[-1]:.4f}  epochs={clf.n_iter_}")
    print(f"conv_W_.shape={clf.conv_W_.shape}  fc_W_.shape={clf.fc_W_.shape}")


if __name__ == "__main__":
    main()
