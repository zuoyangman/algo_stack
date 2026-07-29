"""StandardScaler: zero-mean, unit-variance feature scaling."""

from __future__ import annotations

import numpy as np

from algo_stack._base import BaseEstimator, TransformerMixin
from algo_stack.utils.validation import check_array


class StandardScaler(BaseEstimator, TransformerMixin):
    """Standardise features to zero mean and unit variance.

    Parameters
    ----------
    with_mean : bool, default True
        If True, centre data before scaling.
    with_std : bool, default True
        If True, scale to unit variance.

    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
        Per-feature mean (zeros if ``with_mean=False``).
    scale_ : ndarray of shape (n_features,)
        Per-feature scale (ones if ``with_std=False``). Zero-std features
        are given scale 1.0 to avoid division by zero.
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
        self._check_is_fitted(["mean_", "scale_"])
        X = check_array(X)
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["mean_", "scale_"])
        X = check_array(X)
        return X * self.scale_ + self.mean_

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, y).transform(X)
