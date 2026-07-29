"""K-Nearest-Neighbours classifier & regressor (NumPy).

Supports brute-force, KD-Tree, and Ball-Tree neighbour search.
"""

from __future__ import annotations

import numpy as np

from algo_stack._base import BaseEstimator, ClassifierMixin, RegressorMixin
from algo_stack.utils.neighbors import BallTree, KDTree
from algo_stack.utils.validation import check_array, check_X_y


def _pairwise_sq_euclidean(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared L2 distances via ``||a−b||² = ||a||² + ||b||² − 2 a·b``."""

    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    cross = A @ B.T
    d2 = aa + bb - 2.0 * cross
    np.maximum(d2, 0.0, out=d2)
    return d2


class _KNNBase(BaseEstimator):
    """Shared fit / neighbour-lookup logic for classifier and regressor."""

    def __init__(
        self,
        *,
        n_neighbors: int = 5,
        weights: str = "uniform",
        metric: str = "euclidean",
        algorithm: str = "auto",
        leaf_size: int = 16,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.algorithm = algorithm
        self.leaf_size = leaf_size

    def _validate_init(self) -> None:
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1.")
        if self.weights not in ("uniform", "distance"):
            raise ValueError("weights must be 'uniform' or 'distance'.")
        if self.metric != "euclidean":
            raise ValueError(
                f"metric={self.metric!r} not supported; only 'euclidean' for now."
            )
        if self.algorithm not in ("auto", "brute", "kd_tree", "ball_tree"):
            raise ValueError(
                f"algorithm={self.algorithm!r}; expected "
                "'auto', 'brute', 'kd_tree', or 'ball_tree'."
            )

    def _choose_algorithm(self, n_features: int) -> str:
        if self.algorithm != "auto":
            return self.algorithm
        # Trees help most in low–moderate dimensions.
        return "kd_tree" if n_features <= 15 else "brute"

    def _store(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X_train_ = X
        self.y_train_ = y
        algo = self._choose_algorithm(X.shape[1])
        self.algorithm_ = algo
        if algo == "kd_tree":
            self._tree_ = KDTree(X, leaf_size=self.leaf_size)
        elif algo == "ball_tree":
            self._tree_ = BallTree(X, leaf_size=self.leaf_size)
        else:
            self._tree_ = None

    def _kneighbors(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (distances, indices) of the k nearest training points."""

        self._check_is_fitted(["X_train_"])
        X = check_array(X)
        n_train = self.X_train_.shape[0]
        k = min(self.n_neighbors, n_train)

        if self._tree_ is not None:
            dists, idxs = self._tree_.query(X, k=k, return_distance=True)
            return dists, idxs

        d2 = _pairwise_sq_euclidean(X, self.X_train_)
        part = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        row_idx = np.arange(X.shape[0])[:, None]
        d2_k = d2[row_idx, part]
        order = np.argsort(d2_k, axis=1)
        idx_sorted = part[row_idx, order]
        d_sorted = np.sqrt(d2_k[row_idx, order])
        return d_sorted, idx_sorted


class KNNClassifier(_KNNBase, ClassifierMixin):
    """K-Nearest-Neighbours classifier.

    Parameters
    ----------
    n_neighbors : int, default 5
    weights : {"uniform", "distance"}, default "uniform"
    metric : {"euclidean"}, default "euclidean"
    algorithm : {"auto", "brute", "kd_tree", "ball_tree"}, default "auto"
    leaf_size : int, default 16
        Leaf size for Ball-Tree (ignored by KD-Tree in this reference impl).

    Attributes
    ----------
    classes_ : ndarray
    X_train_ : ndarray of shape (n_train, n_features)
    y_train_ : ndarray of shape (n_train,)
    algorithm_ : str
        Concrete algorithm chosen after ``fit``.
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
        neighbour_labels = self.y_train_[neigh]
        n_query, k = neighbour_labels.shape

        if self.weights == "uniform":
            w = np.ones_like(neighbour_labels, dtype=np.float64)
        else:
            w = 1.0 / (dists + 1e-12)

        proba = np.zeros((n_query, n_classes))
        np.add.at(proba, (np.arange(n_query)[:, None], neighbour_labels), w)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class KNNRegressor(_KNNBase, RegressorMixin):
    """K-Nearest-Neighbours regressor.

    Same constructor as ``KNNClassifier``.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNRegressor":
        self._validate_init()
        X, y = check_X_y(X, y)
        self._store(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        dists, neigh = self._kneighbors(X)
        targets = self.y_train_[neigh]
        if self.weights == "uniform":
            return targets.mean(axis=1)
        w = 1.0 / (dists + 1e-12)
        return np.sum(w * targets, axis=1) / w.sum(axis=1)
