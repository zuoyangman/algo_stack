"""Mini-Batch K-Means clustering (Sculley 2010)."""

from __future__ import annotations

import numpy as np

from algo_stack._base import BaseEstimator, ClusterMixin
from algo_stack.unsupervised.kmeans.kmeans import _kmeans_pp_init, _sq_dist_to_centres
from algo_stack.utils.validation import check_array, check_random_state


class MiniBatchKMeans(BaseEstimator, ClusterMixin):
    """Online / mini-batch K-Means clustering.

    Parameters
    ----------
    n_clusters : int, default 8
    batch_size : int, default 100
        Number of samples drawn uniformly at each iteration.
    n_init : int, default 3
        Independent restarts; the run with lowest inertia wins.
    max_iter : int, default 100
        Number of mini-batch update steps per restart.
    init : {"k-means++", "random"}, default "k-means++"
    random_state : int | Generator | None, default None

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
    labels_ : ndarray of shape (n_samples,)
    inertia_ : float
    n_iter_ : int
    """

    def __init__(
        self,
        *,
        n_clusters: int = 8,
        batch_size: int = 100,
        n_init: int = 3,
        max_iter: int = 100,
        init: str = "k-means++",
        random_state: int | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.n_init = n_init
        self.max_iter = max_iter
        self.init = init
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
    ) -> tuple[np.ndarray, float, int]:
        n_samples = X.shape[0]
        centres = self._init_centres(X, rng)
        counts = np.zeros(self.n_clusters, dtype=np.float64)
        batch = min(self.batch_size, n_samples)

        for it in range(self.max_iter):
            idx = rng.choice(n_samples, size=batch, replace=False)
            Xb = X[idx]
            d2 = _sq_dist_to_centres(Xb, centres)
            labels_b = np.argmin(d2, axis=1)

            for j in range(batch):
                c = int(labels_b[j])
                counts[c] += 1.0
                eta = 1.0 / counts[c]
                centres[c] = (1.0 - eta) * centres[c] + eta * Xb[j]

            # Re-seed empty clusters toward farthest points in the batch.
            empty = np.where(counts == 0)[0]
            if empty.size:
                closest = np.min(d2, axis=1)
                for c in empty:
                    far = int(np.argmax(closest))
                    centres[c] = Xb[far].copy()
                    counts[c] = 1.0
                    closest[far] = -np.inf

        d2_full = _sq_dist_to_centres(X, centres)
        labels = np.argmin(d2_full, axis=1)
        # Final mean update with full data for empty / drifted centres.
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.any():
                centres[k] = X[mask].mean(axis=0)
            else:
                far_idx = int(np.argmax(np.min(d2_full, axis=1)))
                centres[k] = X[far_idx]
                labels[far_idx] = k

        d2_full = _sq_dist_to_centres(X, centres)
        labels = np.argmin(d2_full, axis=1)
        inertia = float(np.sum(d2_full[np.arange(n_samples), labels]))
        return centres, inertia, it + 1

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "MiniBatchKMeans":
        if self.n_clusters < 1:
            raise ValueError("n_clusters must be >= 1.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        X = check_array(X)
        if X.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_samples={X.shape[0]} < n_clusters={self.n_clusters}."
            )

        rng = check_random_state(self.random_state)
        best_centres: np.ndarray | None = None
        best_inertia = np.inf
        best_n_iter = 0
        for _ in range(self.n_init):
            run_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
            centres, inertia, n_iter = self._single_run(X, run_rng)
            if inertia < best_inertia:
                best_centres = centres
                best_inertia = inertia
                best_n_iter = n_iter

        assert best_centres is not None
        self.cluster_centers_ = best_centres
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        d2 = _sq_dist_to_centres(X, self.cluster_centers_)
        self.labels_ = np.argmin(d2, axis=1)
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
