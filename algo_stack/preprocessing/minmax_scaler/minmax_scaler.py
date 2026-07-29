"""MinMaxScaler: scale features to a given range."""

from __future__ import annotations

import numpy as np

from algo_stack._base import BaseEstimator, TransformerMixin
from algo_stack.utils.validation import check_array


class MinMaxScaler(BaseEstimator, TransformerMixin):
    """Scale features to a fixed range, typically ``[0, 1]``.

    The transform is::

        X_scaled = X * scale_ + min_

    where ``scale_`` and ``min_`` are chosen so that each feature maps
    ``[data_min_, data_max_]`` onto ``feature_range``.

    Parameters
    ----------
    feature_range : tuple of float, default (0, 1)
        Desired ``(min, max)`` of transformed data.

    Attributes
    ----------
    min_ : ndarray of shape (n_features,)
        Additive offset applied after scaling.
    scale_ : ndarray of shape (n_features,)
        Per-feature multiplicative scale.
    data_min_ : ndarray of shape (n_features,)
        Per-feature minimum seen in ``fit``.
    data_max_ : ndarray of shape (n_features,)
        Per-feature maximum seen in ``fit``.
    """

    def __init__(self, *, feature_range: tuple[float, float] = (0, 1)) -> None:
        self.feature_range = feature_range

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "MinMaxScaler":
        X = check_array(X)
        fr_min, fr_max = self.feature_range
        if fr_min >= fr_max:
            raise ValueError(
                f"Minimum of feature range must be smaller than maximum; "
                f"got {self.feature_range}."
            )

        self.data_min_ = X.min(axis=0)
        self.data_max_ = X.max(axis=0)
        data_range = self.data_max_ - self.data_min_
        data_range[data_range == 0] = 1.0

        self.scale_ = (fr_max - fr_min) / data_range
        self.min_ = fr_min - self.data_min_ * self.scale_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["scale_", "min_"])
        X = check_array(X)
        return X * self.scale_ + self.min_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["scale_", "min_"])
        X = check_array(X)
        return (X - self.min_) / self.scale_

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, y).transform(X)
