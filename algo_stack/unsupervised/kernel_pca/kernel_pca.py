"""Kernel Principal Component Analysis."""

from __future__ import annotations

import numpy as np

from algo_stack._base import BaseEstimator, TransformerMixin
from algo_stack.utils.validation import check_array


def _pairwise_sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    d2 = aa + bb - 2.0 * (A @ B.T)
    np.maximum(d2, 0.0, out=d2)
    return d2


def _kernel_matrix(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    kernel: str,
    gamma: float | None,
    degree: int,
) -> np.ndarray:
    if kernel == "linear":
        return X @ Y.T
    g = 1.0 / X.shape[1] if gamma is None else float(gamma)
    if kernel == "rbf":
        return np.exp(-g * _pairwise_sq_dists(X, Y))
    if kernel == "poly":
        return (g * (X @ Y.T) + 1.0) ** degree
    raise ValueError(f"Unknown kernel={kernel!r}")


def _center_kernel(K: np.ndarray) -> np.ndarray:
    """Double-centre a square Gram matrix: K̃ = H K H."""

    n = K.shape[0]
    row_mean = K.mean(axis=1, keepdims=True)
    col_mean = K.mean(axis=0, keepdims=True)
    return K - row_mean - col_mean + K.mean()


class KernelPCA(BaseEstimator, TransformerMixin):
    """Non-linear PCA in a reproducing kernel Hilbert space.

    Parameters
    ----------
    n_components : int | None, default None
    kernel : {"rbf", "linear", "poly"}, default "rbf"
    gamma : float | None, default None
        Kernel coefficient for RBF / poly. ``None`` → ``1 / n_features``.
    degree : int, default 3
        Degree of the polynomial kernel.

    Attributes
    ----------
    alphas_ : ndarray of shape (n_samples, n_components)
        Eigenvectors of the centred kernel matrix (scaled).
    lambdas_ : ndarray of shape (n_components,)
        Eigenvalues of the centred kernel matrix.
    X_fit_ : ndarray of shape (n_samples, n_features)
    n_components_ : int
    """

    def __init__(
        self,
        *,
        n_components: int | None = None,
        kernel: str = "rbf",
        gamma: float | None = None,
        degree: int = 3,
    ) -> None:
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "KernelPCA":
        X = check_array(X)
        n_samples = X.shape[0]
        n_comp = n_samples if self.n_components is None else int(self.n_components)
        if n_comp < 1 or n_comp > n_samples:
            raise ValueError(
                f"n_components must be in [1, {n_samples}], got {n_comp}."
            )

        K = _kernel_matrix(
            X, X, kernel=self.kernel, gamma=self.gamma, degree=self.degree
        )
        Kc = _center_kernel(K)
        eigvals, eigvecs = np.linalg.eigh(Kc)
        # eigh returns ascending order — take largest.
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Drop non-positive eigenvalues for numerical stability.
        pos = eigvals > 1e-12
        eigvals = eigvals[pos]
        eigvecs = eigvecs[:, pos]
        n_comp = min(n_comp, eigvals.shape[0])
        if n_comp < 1:
            raise RuntimeError("Kernel matrix has no positive eigenvalues.")

        lambdas = eigvals[:n_comp]
        alphas = eigvecs[:, :n_comp] / np.sqrt(lambdas)

        self.X_fit_ = X.copy()
        self.alphas_ = alphas
        self.lambdas_ = lambdas
        self.n_components_ = n_comp
        # Needed to centre out-of-sample kernels.
        self._K_fit_row_mean_ = K.mean(axis=1)
        self._K_fit_mean_ = float(K.mean())
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["alphas_", "X_fit_"])
        X = check_array(X)
        K = _kernel_matrix(
            X,
            self.X_fit_,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
        )
        # Centre using training statistics: K̃ = K - 1μᵀ - μ1ᵀ + μ̄
        Kc = K - self._K_fit_row_mean_[None, :] - K.mean(axis=1, keepdims=True)
        Kc = Kc + self._K_fit_mean_
        return Kc @ self.alphas_

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        self.fit(X, y)
        # In-sample projection: √λ · α scaled already as Kc @ (v/√λ) = √λ v
        K = _kernel_matrix(
            self.X_fit_,
            self.X_fit_,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
        )
        Kc = _center_kernel(K)
        return Kc @ self.alphas_
