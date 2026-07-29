"""DBSCAN density-based clustering (Ester et al. 1996)."""

from __future__ import annotations

import numpy as np

from algo_stack._base import BaseEstimator, ClusterMixin
from algo_stack.utils.validation import check_array


def _pairwise_distances(X: np.ndarray, metric: str) -> np.ndarray:
    if metric == "euclidean":
        xx = np.sum(X * X, axis=1)[:, None]
        d2 = xx + xx.T - 2.0 * (X @ X.T)
        np.maximum(d2, 0.0, out=d2)
        return np.sqrt(d2)
    if metric == "manhattan":
        # Brute-force L1 for small toy problems.
        n = X.shape[0]
        D = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            D[i] = np.sum(np.abs(X - X[i]), axis=1)
        return D
    raise ValueError(f"Unsupported metric={metric!r}")


class DBSCAN(BaseEstimator, ClusterMixin):
    """Density-Based Spatial Clustering of Applications with Noise.

    Parameters
    ----------
    eps : float, default 0.5
        Neighbourhood radius.
    min_samples : int, default 5
        Minimum neighbours (including self) to be a core point.
    metric : {"euclidean", "manhattan"}, default "euclidean"

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels; ``-1`` denotes noise.
    core_sample_indices_ : ndarray of shape (n_core_samples,)
        Indices of core samples in the training set.
    """

    def __init__(
        self,
        *,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
    ) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "DBSCAN":
        if self.eps <= 0:
            raise ValueError("eps must be > 0.")
        if self.min_samples < 1:
            raise ValueError("min_samples must be >= 1.")
        X = check_array(X)
        n = X.shape[0]
        D = _pairwise_distances(X, self.metric)
        neighbours = [np.where(D[i] <= self.eps)[0] for i in range(n)]
        is_core = np.array(
            [len(neighbours[i]) >= self.min_samples for i in range(n)], dtype=bool
        )

        labels = np.full(n, -1, dtype=np.int64)
        cluster_id = 0
        for i in range(n):
            if labels[i] != -1 or not is_core[i]:
                continue
            # Expand cluster from core point i (BFS).
            stack = [i]
            labels[i] = cluster_id
            while stack:
                q = stack.pop()
                if not is_core[q]:
                    continue
                for j in neighbours[q]:
                    if labels[j] == -1:
                        labels[j] = cluster_id
                        if is_core[j]:
                            stack.append(int(j))
            cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.where(is_core)[0]
        return self

    def fit_predict(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        self.fit(X, y)
        return self.labels_
