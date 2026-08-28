"""K-Nearest-Neighbours classifier & regressor (brute-force, NumPy).

The brute-force `O(n_train · n_query · d)` distance computation is the
clearest reference implementation. See ``EXTENSION.md`` for KD-tree / Ball-tree
acceleration ideas.
"""

from __future__ import annotations

import numpy as np

from algo_stack._base import BaseEstimator, ClassifierMixin, RegressorMixin
from algo_stack.utils.validation import check_array, check_X_y


def _pairwise_sq_euclidean(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return the matrix of squared L2 distances between rows of A and B.

    Uses the identity ``||a − b||² = ||a||² + ||b||² − 2 a·b`` so we only
    issue one matrix multiplication. Negative values caused by floating
    point are clipped to 0 before sqrt is taken.
    """

    aa = np.sum(A * A, axis=1)[:, None]      # (n_a, 1)
    bb = np.sum(B * B, axis=1)[None, :]      # (1, n_b)
    cross = A @ B.T                          # (n_a, n_b)
    d2 = aa + bb - 2.0 * cross
    np.maximum(d2, 0.0, out=d2)
    return d2


class _KNNBase(BaseEstimator):
    """Shared fit/neighbour-lookup logic for the classifier and regressor."""

    def __init__(
        self,
        *,
        n_neighbors: int = 5,
        weights: str = "uniform",
        metric: str = "euclidean",
    ) -> None:
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric

    def _validate_init(self) -> None:
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1.")
        if self.weights not in ("uniform", "distance"):
            raise ValueError("weights must be 'uniform' or 'distance'.")
        if self.metric != "euclidean":
            raise ValueError(
                f"metric={self.metric!r} not supported; only 'euclidean' for now."
            )

    def _store(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X_train_ = X
        self.y_train_ = y

    def _kneighbors(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (distances, indices) of the k nearest training points."""

        self._check_is_fitted(["X_train_"])
        X = check_array(X)
        n_train = self.X_train_.shape[0]
        k = min(self.n_neighbors, n_train)

        d2 = _pairwise_sq_euclidean(X, self.X_train_)
        # argpartition gives the k smallest unsorted; then we sort just those k.
        part = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        row_idx = np.arange(X.shape[0])[:, None]
        d2_k = d2[row_idx, part]
        order = np.argsort(d2_k, axis=1)
        idx_sorted = part[row_idx, order]
        d_sorted = np.sqrt(d2_k[row_idx, order])
        return d_sorted, idx_sorted


class KNNClassifier(_KNNBase, ClassifierMixin):
    """K-Nearest-Neighbours classifier.

    Attributes
    ----------
    classes_ : ndarray
    X_train_ : ndarray of shape (n_train, n_features)
    y_train_ : ndarray of shape (n_train,) (integer-encoded class indices)
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        self._validate_init()
        X, y = check_X_y(X, y, y_numeric=False)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self._store(X, y_idx)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        dists, neigh = self._kneighbors(X)
        n_classes = len(self.classes_)
        neighbour_labels = self.y_train_[neigh]  # (n_query, k)
        n_query, k = neighbour_labels.shape

        if self.weights == "uniform":
            w = np.ones_like(neighbour_labels, dtype=np.float64)
        else:  # "distance"
            # Inverse distance, with eps for stability if duplicates exist.
            w = 1.0 / (dists + 1e-12)

        proba = np.zeros((n_query, n_classes))
        # vectorised scatter-add along axis=1
        np.add.at(proba, (np.arange(n_query)[:, None], neighbour_labels), w)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class KNNRegressor(_KNNBase, RegressorMixin):
    """K-Nearest-Neighbours regressor.

    Attributes
    ----------
    X_train_ : ndarray of shape (n_train, n_features)
    y_train_ : ndarray of shape (n_train,)
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNRegressor":
        self._validate_init()
        X, y = check_X_y(X, y)
        self._store(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        dists, neigh = self._kneighbors(X)
        targets = self.y_train_[neigh]  # (n_query, k)
        if self.weights == "uniform":
            return targets.mean(axis=1)
        w = 1.0 / (dists + 1e-12)
        return np.sum(w * targets, axis=1) / w.sum(axis=1)
