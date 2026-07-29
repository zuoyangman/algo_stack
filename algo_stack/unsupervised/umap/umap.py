"""Simplified didactic UMAP (Uniform Manifold Approximation and Projection).

Dense fuzzy-graph construction + SGD layout for small ``n`` (tests use ≤ 100).
"""

from __future__ import annotations

import numpy as np

from algo_stack._base import BaseEstimator, TransformerMixin
from algo_stack.utils.validation import check_array, check_random_state


def _pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    xx = np.sum(X * X, axis=1)[:, None]
    d2 = xx + xx.T - 2.0 * (X @ X.T)
    np.maximum(d2, 0.0, out=d2)
    np.fill_diagonal(d2, 0.0)
    return d2


def _find_sigma(distances: np.ndarray, n_neighbors: int) -> tuple[float, float]:
    """Binary search σ for one point's knn distances (exclude self already)."""

    rho = float(max(distances[0], 0.0))
    target = np.log2(max(float(n_neighbors), 2.0))
    lo, hi = 1e-10, 1e3
    sigma = 1.0
    for _ in range(64):
        psum = 0.0
        for d in distances:
            val = float(d) - rho
            psum += 1.0 if val <= 0.0 else float(np.exp(-val / sigma))
        if abs(psum - target) < 1e-5:
            break
        if psum > target:
            hi = sigma
            sigma = 0.5 * (lo + hi)
        else:
            lo = sigma
            sigma = 0.5 * (lo + hi)
    return rho, float(max(sigma, 1e-10))


def _fuzzy_simplicial_set(
    X: np.ndarray, n_neighbors: int
) -> np.ndarray:
    """Return dense fuzzy membership matrix ``B`` (symmetric set union)."""

    n = X.shape[0]
    k = min(n_neighbors, n - 1)
    if k < 1:
        raise ValueError("Need at least 2 samples to build a knn graph.")
    d2 = _pairwise_sq_dists(X)
    d = np.sqrt(d2)

    # Directed fuzzy memberships.
    mu = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        # Indices of k nearest (exclude self).
        order = np.argsort(d[i])
        knn_idx = order[1 : k + 1]
        knn_dist = d[i, knn_idx]
        rho, sigma = _find_sigma(knn_dist, k)
        for j, dist in zip(knn_idx, knn_dist):
            val = float(dist) - rho
            mu[i, j] = 1.0 if val <= 0.0 else float(np.exp(-val / sigma))

    # Fuzzy set union symmetrisation (UMAP): a ∪ b = a + b − a·b.
    B = mu + mu.T - mu * mu.T
    np.fill_diagonal(B, 0.0)
    return B


def _fit_ab(min_dist: float) -> tuple[float, float]:
    """Approximate the standard UMAP (a, b) curve for a given ``min_dist``.

    Fits ``1 / (1 + a x^{2b}) ≈`` the piecewise smooth membership used in the
    paper, via a tiny alternating least-squares solve on a grid.
    """

    xs = np.linspace(0.0, 3.0, 300)
    y = np.ones_like(xs)
    mask = xs > min_dist
    y[mask] = np.exp(-(xs[mask] - min_dist))

    x_fit = xs[mask]
    y_fit = np.clip(y[mask], 1e-8, 1.0)
    a, b = 1.0, 1.0
    for _ in range(40):
        t = x_fit ** (2.0 * b)
        target = (1.0 / y_fit) - 1.0
        a = float(np.maximum(np.sum(t * target) / (np.sum(t * t) + 1e-12), 1e-8))
        best_b, best_err = b, np.inf
        for b_try in np.linspace(max(0.1, b - 0.3), b + 0.3, 13):
            pred = 1.0 / (1.0 + a * (x_fit ** (2.0 * b_try)))
            err = float(np.mean((pred - y_fit) ** 2))
            if err < best_err:
                best_err, best_b = err, float(b_try)
        b = best_b
    return a, float(b)


def _spectral_init(B: np.ndarray, n_components: int, rng: np.random.Generator) -> np.ndarray:
    """Laplacian eigenmaps init; fall back to random on numerical failure."""

    n = B.shape[0]
    deg = B.sum(axis=1)
    # Normalised Laplacian L = I - D^{-1/2} B D^{-1/2}.
    d_inv_sqrt = np.zeros(n)
    nz = deg > 0
    d_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    D_half = np.diag(d_inv_sqrt)
    L = np.eye(n) - D_half @ B @ D_half
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Smallest non-trivial eigenvectors.
        order = np.argsort(eigvals)
        comps = []
        for idx in order:
            if eigvals[idx] < 1e-10:
                continue
            comps.append(eigvecs[:, idx])
            if len(comps) >= n_components:
                break
        if len(comps) < n_components:
            raise np.linalg.LinAlgError("insufficient eigenvectors")
        Y = np.column_stack(comps)
        Y = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-8)
        return Y.astype(np.float64)
    except np.linalg.LinAlgError:
        return rng.normal(scale=1e-4, size=(n, n_components))


def _optimize_layout(
    B: np.ndarray,
    Y: np.ndarray,
    *,
    a: float,
    b: float,
    n_epochs: int,
    learning_rate: float,
    rng: np.random.Generator,
    n_neg: int = 5,
) -> np.ndarray:
    """SGD on fuzzy cross-entropy (attraction on edges + random repulsion)."""

    n, dim = Y.shape
    # Edge list with weights above a tiny threshold.
    ii, jj = np.where(np.triu(B, k=1) > 1e-8)
    weights = B[ii, jj]
    if len(weights) == 0:
        return Y

    # Epochs with linearly decaying learning rate.
    for epoch in range(n_epochs):
        alpha = learning_rate * (1.0 - epoch / max(n_epochs, 1))
        # Shuffle edges.
        order = rng.permutation(len(weights))
        for e in order:
            i, j = int(ii[e]), int(jj[e])
            w = float(weights[e])
            diff = Y[i] - Y[j]
            dist_sq = float(np.dot(diff, diff))
            if dist_sq < 1e-16:
                dist_sq = 1e-16
            # Attractive force (umap-learn style).
            grad_coeff = (
                -2.0 * a * b * (dist_sq ** (b - 1.0)) / (1.0 + a * (dist_sq**b))
            )
            grad = np.clip(grad_coeff * diff, -4.0, 4.0)
            Y[i] += alpha * w * grad
            Y[j] -= alpha * w * grad

            # Negative samples (repulsion).
            for _ in range(n_neg):
                k = int(rng.integers(0, n))
                if k == i:
                    continue
                diff_n = Y[i] - Y[k]
                dist_sq_n = float(np.dot(diff_n, diff_n))
                if dist_sq_n < 1e-16:
                    dist_sq_n = 1e-16
                grad_coeff_n = (
                    2.0 * b / ((0.001 + dist_sq_n) * (1.0 + a * (dist_sq_n**b)))
                )
                # Weight repulsion lightly (1 − w_ik ≈ 1 for non-edges).
                w_ik = float(B[i, k])
                grad_n = np.clip(grad_coeff_n * diff_n, -4.0, 4.0)
                Y[i] += alpha * (1.0 - w_ik) * grad_n

        Y -= Y.mean(axis=0)
    return Y


class UMAP(BaseEstimator, TransformerMixin):
    """Simplified Uniform Manifold Approximation and Projection.

    Builds a fuzzy knn simplicial set, initialises with Laplacian eigenmaps
    (random fallback), then optimises low-d cross-entropy with SGD.

    Parameters
    ----------
    n_components : int, default 2
    n_neighbors : int, default 15
    min_dist : float, default 0.1
    n_epochs : int, default 200
    learning_rate : float, default 1.0
    random_state : int | None, default None
    init : {"spectral", "random"}, default "spectral"

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
    graph_ : ndarray of shape (n_samples, n_samples)
        Fuzzy membership matrix used for layout.
    a_, b_ : float
        Low-d curve parameters fit from ``min_dist``.
    n_features_in_ : int
    """

    def __init__(
        self,
        *,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        n_epochs: int = 200,
        learning_rate: float = 1.0,
        random_state: int | None = None,
        init: str = "spectral",
    ) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.init = init

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        X = check_array(X)
        n = X.shape[0]
        if self.n_components < 1:
            raise ValueError("n_components must be >= 1.")
        if self.n_neighbors < 2:
            raise ValueError("n_neighbors must be >= 2.")
        if n < 3:
            raise ValueError("UMAP requires at least 3 samples.")
        if self.min_dist < 0:
            raise ValueError("min_dist must be >= 0.")
        if self.init not in ("spectral", "random"):
            raise ValueError("init must be 'spectral' or 'random'.")

        rng = check_random_state(self.random_state)
        B = _fuzzy_simplicial_set(X, self.n_neighbors)
        a, b = _fit_ab(self.min_dist)

        if self.init == "spectral":
            Y = _spectral_init(B, self.n_components, rng)
        else:
            Y = rng.normal(scale=1e-4, size=(n, self.n_components))

        Y = _optimize_layout(
            B,
            Y,
            a=a,
            b=b,
            n_epochs=self.n_epochs,
            learning_rate=self.learning_rate,
            rng=rng,
        )

        self.embedding_ = Y
        self.graph_ = B
        self.a_ = a
        self.b_ = b
        self.n_features_in_ = X.shape[1]
        self._n_samples_fit_ = n
        return Y.copy()

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "UMAP":
        self.fit_transform(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return the training embedding.

        Out-of-sample embedding is **not** implemented. If ``X`` has the same
        number of rows as the training set, ``embedding_`` is returned (the
        training map). Otherwise ``NotImplementedError`` is raised.
        """

        self._check_is_fitted(["embedding_"])
        X = check_array(X)
        if X.shape[0] != self._n_samples_fit_:
            raise NotImplementedError(
                "Out-of-sample UMAP transform is not implemented in this "
                "didactic version; use fit_transform on the full dataset or "
                "access `.embedding_` after fit."
            )
        return self.embedding_.copy()
