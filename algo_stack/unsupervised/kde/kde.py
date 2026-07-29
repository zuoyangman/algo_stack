"""Kernel Density Estimation."""

from __future__ import annotations

import numpy as np

from algo_stack._base import BaseEstimator
from algo_stack.utils.validation import check_array


def _pairwise_sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    d2 = aa + bb - 2.0 * (A @ B.T)
    np.maximum(d2, 0.0, out=d2)
    return d2


class KernelDensity(BaseEstimator):
    """Non-parametric density estimation with a fixed-bandwidth kernel.

    Parameters
    ----------
    bandwidth : float, default 1.0
        Kernel bandwidth ``h > 0``.
    kernel : {"gaussian"}, default "gaussian"

    Attributes
    ----------
    X_fit_ : ndarray of shape (n_samples, n_features)
    n_features_in_ : int
    """

    def __init__(self, *, bandwidth: float = 1.0, kernel: str = "gaussian") -> None:
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "KernelDensity":
        if self.bandwidth <= 0:
            raise ValueError("bandwidth must be > 0.")
        if self.kernel != "gaussian":
            raise ValueError(f"Unsupported kernel={self.kernel!r}")
        X = check_array(X)
        self.X_fit_ = X.copy()
        self.n_features_in_ = X.shape[1]
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the log density model on the data ``X``."""

        self._check_is_fitted(["X_fit_"])
        X = check_array(X)
        n_samples, n_features = self.X_fit_.shape
        h = self.bandwidth
        d2 = _pairwise_sq_dists(X, self.X_fit_)
        # log φ((x-x_i)/h) = -0.5 ||u||² - d/2 log(2π) - d log h
        log_kernel = -0.5 * d2 / (h * h)
        log_norm = -0.5 * n_features * np.log(2.0 * np.pi) - n_features * np.log(h)
        log_kernel = log_kernel + log_norm
        # log (1/n Σ k_i) via logsumexp
        m = np.max(log_kernel, axis=1, keepdims=True)
        log_sum = m.ravel() + np.log(np.sum(np.exp(log_kernel - m), axis=1))
        return log_sum - np.log(n_samples)

    def score(self, X: np.ndarray, y: np.ndarray | None = None) -> float:
        """Mean log-likelihood of the samples in ``X``."""

        return float(np.mean(self.score_samples(X)))
