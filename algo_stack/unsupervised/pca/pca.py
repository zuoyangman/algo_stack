"""Principal Component Analysis via SVD."""

from __future__ import annotations

import numpy as np

from algo_stack._base import BaseEstimator, TransformerMixin
from algo_stack.utils.validation import check_array


class PCA(BaseEstimator, TransformerMixin):
    """Linear dimensionality reduction using truncated SVD of centred data.

    Parameters
    ----------
    n_components : int | None, default None
        Number of components to keep. ``None`` keeps ``min(n_samples, n_features)``.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal axes in feature space (rows are eigenvectors).
    explained_variance_ : ndarray of shape (n_components,)
    explained_variance_ratio_ : ndarray of shape (n_components,)
    mean_ : ndarray of shape (n_features,)
    n_components_ : int
    """

    def __init__(self, *, n_components: int | None = None) -> None:
        self.n_components = n_components

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "PCA":
        X = check_array(X)
        n_samples, n_features = X.shape
        max_comp = min(n_samples, n_features)
        n_comp = max_comp if self.n_components is None else int(self.n_components)
        if n_comp < 1 or n_comp > max_comp:
            raise ValueError(
                f"n_components must be in [1, {max_comp}], got {n_comp}."
            )

        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        # Economy SVD: Xc = U S Vt
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.components_ = Vt[:n_comp]
        # Population-style explained variance (sklearn uses n_samples - 1).
        explained = (S**2) / max(n_samples - 1, 1)
        self.explained_variance_ = explained[:n_comp]
        total = explained.sum()
        if total > 0:
            self.explained_variance_ratio_ = self.explained_variance_ / total
        else:
            self.explained_variance_ratio_ = np.zeros(n_comp)
        self.n_components_ = n_comp
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["components_", "mean_"])
        X = check_array(X)
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["components_", "mean_"])
        X = check_array(X)
        return X @ self.components_ + self.mean_

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, y).transform(X)
