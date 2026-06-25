"""K-Means clustering with Lloyd's algorithm + k-means++ initialisation."""

from __future__ import annotations

import numpy as np

from algo_stack._base import BaseEstimator, ClusterMixin
from algo_stack.utils.validation import check_array, check_random_state


def _sq_dist_to_centres(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Squared L2 distance between every row of ``X`` and every row of ``C``."""

    xx = np.sum(X * X, axis=1)[:, None]
    cc = np.sum(C * C, axis=1)[None, :]
    cross = X @ C.T
    d2 = xx + cc - 2.0 * cross
    np.maximum(d2, 0.0, out=d2)
    return d2


def _kmeans_pp_init(
    X: np.ndarray, n_clusters: int, rng: np.random.Generator
) -> np.ndarray:
    """k-means++ seeding (Arthur & Vassilvitskii 2007)."""

    n_samples = X.shape[0]
    first = int(rng.integers(0, n_samples))
    centres = [X[first]]
    closest_sq = _sq_dist_to_centres(X, X[first : first + 1]).ravel()
    for _ in range(1, n_clusters):
        total = closest_sq.sum()
        if total <= 0:
            # All points coincide with chosen centres; pick uniformly.
            idx = int(rng.integers(0, n_samples))
        else:
            probs = closest_sq / total
            idx = int(rng.choice(n_samples, p=probs))
        centres.append(X[idx])
        new_d2 = _sq_dist_to_centres(X, X[idx : idx + 1]).ravel()
        closest_sq = np.minimum(closest_sq, new_d2)
    return np.array(centres)


class KMeans(BaseEstimator, ClusterMixin):
    """K-Means clustering via Lloyd's algorithm.

    Parameters
    ----------
    n_clusters : int, default 8
    init : {"k-means++", "random"}, default "k-means++"
    n_init : int, default 10
        How many independent restarts; the best inertia wins.
    max_iter : int, default 300
    tol : float, default 1e-4
        Relative Frobenius-norm change in centroids below which we declare
        convergence.
    random_state : int | Generator | None, default None

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
    labels_ : ndarray of shape (n_samples,)
    inertia_ : float
        Sum of squared distances from each sample to its assigned centroid.
    n_iter_ : int
        Iterations of the best run.
    """

    def __init__(
        self,
        *,
        n_clusters: int = 8,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _init_centres(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.init == "k-means++":
            return _kmeans_pp_init(X, self.n_clusters, rng)
        if self.init == "random":
            idx = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
            return X[idx].copy()
        raise ValueError(f"Unknown init={self.init!r}")

    def _single_run(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        centres = self._init_centres(X, rng)
        labels = np.zeros(X.shape[0], dtype=np.int64)
        for it in range(self.max_iter):
            d2 = _sq_dist_to_centres(X, centres)
            new_labels = np.argmin(d2, axis=1)
            new_centres = np.empty_like(centres)
            for k in range(self.n_clusters):
                mask = new_labels == k
                if mask.any():
                    new_centres[k] = X[mask].mean(axis=0)
                else:
                    # Empty cluster: re-seed to the point farthest from its centre.
                    far_idx = int(np.argmax(np.min(d2, axis=1)))
                    new_centres[k] = X[far_idx]

            shift = float(np.linalg.norm(new_centres - centres))
            base = float(np.linalg.norm(centres)) + 1e-12
            centres = new_centres
            labels = new_labels
            if shift / base < self.tol:
                break

        d2 = _sq_dist_to_centres(X, centres)
        labels = np.argmin(d2, axis=1)
        inertia = float(np.sum(d2[np.arange(X.shape[0]), labels]))
        return centres, labels, inertia, it + 1

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "KMeans":
        if self.n_clusters < 1:
            raise ValueError("n_clusters must be >= 1.")
        X = check_array(X)
        if X.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_samples={X.shape[0]} < n_clusters={self.n_clusters}."
            )

        rng = check_random_state(self.random_state)
        best: tuple[np.ndarray, np.ndarray, float, int] | None = None
        for _ in range(self.n_init):
            run_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
            result = self._single_run(X, run_rng)
            if best is None or result[2] < best[2]:
                best = result

        assert best is not None
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = best
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["cluster_centers_"])
        X = check_array(X)
        d2 = _sq_dist_to_centres(X, self.cluster_centers_)
        return np.argmin(d2, axis=1)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return the distance from each sample to every centroid."""

        self._check_is_fitted(["cluster_centers_"])
        X = check_array(X)
        return np.sqrt(_sq_dist_to_centres(X, self.cluster_centers_))
