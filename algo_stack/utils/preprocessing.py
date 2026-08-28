"""Tiny preprocessing helpers used by examples and tests.

These are *minimal* implementations sufficient for the bundled demos; a fuller
``preprocessing`` package is planned in ``docs/ROADMAP.md``.
"""

from __future__ import annotations

import numpy as np

from algo_stack.utils.validation import check_array, check_random_state


class StandardScaler:
    """Standardise features to zero mean and unit variance.

    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
    scale_ : ndarray of shape (n_features,)
    """

    def __init__(self, *, with_mean: bool = True, with_std: bool = True) -> None:
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "StandardScaler":
        X = check_array(X)
        self.mean_ = X.mean(axis=0) if self.with_mean else np.zeros(X.shape[1])
        if self.with_std:
            std = X.std(axis=0)
            std[std == 0] = 1.0
            self.scale_ = std
        else:
            self.scale_ = np.ones(X.shape[1])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X)
        return X * self.scale_ + self.mean_


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.25,
    random_state: int | np.random.Generator | None = None,
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle and split arrays into train / test partitions."""

    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1).")

    rng = check_random_state(random_state)
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    n_test = int(round(test_size * n))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
