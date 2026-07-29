"""Shared kernel functions for SVM / Kernel Ridge (NumPy only)."""

from __future__ import annotations

from typing import Any

import numpy as np


def pairwise_sq_euclidean(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared L2 distances between rows of ``A`` and ``B``."""

    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    d2 = aa + bb - 2.0 * (A @ B.T)
    np.maximum(d2, 0.0, out=d2)
    return d2


def resolve_gamma(
    gamma: Any,
    X: np.ndarray,
) -> float:
    """Resolve ``gamma`` for RBF / polynomial kernels.

    ``"scale"`` → ``1 / (n_features * X.var())`` (sklearn convention).
    ``"auto"``  → ``1 / n_features``.
    """

    if isinstance(gamma, str):
        if gamma == "scale":
            var = float(X.var())
            return 1.0 / (X.shape[1] * var) if var > 0.0 else 1.0
        if gamma == "auto":
            return 1.0 / X.shape[1]
        raise ValueError(f"Unknown gamma string {gamma!r}; use 'scale', 'auto', or float.")
    return float(gamma)


def kernel_matrix(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    *,
    kernel: str = "rbf",
    gamma: float = 1.0,
    degree: int = 3,
    coef0: float = 1.0,
) -> np.ndarray:
    """Compute the Gram matrix ``K(X, Y)``.

    Parameters
    ----------
    kernel : {"linear", "rbf", "polynomial", "poly"}
    gamma : float
        Bandwidth for RBF / scale for polynomial (already resolved).
    degree : int
        Polynomial degree.
    coef0 : float
        Independent term in polynomial kernel.
    """

    if Y is None:
        Y = X
    if kernel == "linear":
        return X @ Y.T
    if kernel == "rbf":
        return np.exp(-gamma * pairwise_sq_euclidean(X, Y))
    if kernel in ("polynomial", "poly"):
        return (gamma * (X @ Y.T) + coef0) ** degree
    raise ValueError(
        f"Unknown kernel {kernel!r}; expected 'linear', 'rbf', or 'polynomial'."
    )
