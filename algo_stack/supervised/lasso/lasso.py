"""Lasso regression (OLS + L1) via coordinate descent soft-thresholding.

See ``PRINCIPLE.md`` for the derivation and ``EXTENSION.md`` for variants.
"""

from __future__ import annotations

import numpy as np

from algo_stack._base import BaseEstimator, RegressorMixin
from algo_stack.utils.validation import check_array, check_X_y


def _soft_threshold(z: float, gamma: float) -> float:
    """Soft-threshold operator S(z, γ) = sign(z) · max(|z| − γ, 0)."""
    if z > gamma:
        return z - gamma
    if z < -gamma:
        return z + gamma
    return 0.0


class Lasso(BaseEstimator, RegressorMixin):
    """Lasso regression: ordinary least squares with an L1 weight penalty.

    Minimises

        (1 / (2 n)) ||X w + b − y||²  +  α ||w||₁

    by cyclic coordinate descent with soft-thresholding.  The intercept is
    **not** penalised.

    Parameters
    ----------
    alpha : float, default 1.0
        L1 regularisation strength. Must be ≥ 0.
    max_iter : int, default 1000
        Maximum number of full coordinate sweeps.
    tol : float, default 1e-4
        Stop when the max absolute change in any weight falls below ``tol``.
    fit_intercept : bool, default True
        Whether to estimate an intercept term.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
    intercept_ : float
    n_iter_ : int
        Number of coordinate sweeps actually run.
    """

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-4,
        fit_intercept: bool = True,
    ) -> None:
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Lasso":
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}.")
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape

        if self.fit_intercept:
            X_mean = X.mean(axis=0)
            y_mean = float(y.mean())
            Xc = X - X_mean
            yc = y - y_mean
        else:
            X_mean = np.zeros(n_features)
            y_mean = 0.0
            Xc = np.asarray(X, dtype=np.float64)
            yc = np.asarray(y, dtype=np.float64)

        # Precompute column norms ||X_j||² (constant under cyclic CD).
        col_norm_sq = np.sum(Xc * Xc, axis=0)
        # Avoid division by zero for constant (after centring) columns.
        col_norm_sq = np.where(col_norm_sq < 1e-12, 1.0, col_norm_sq)

        w = np.zeros(n_features)
        # Residual r = yc − Xc w; start at yc since w = 0.
        residual = yc.copy()
        threshold = self.alpha * n_samples  # because objective has 1/(2n)

        n_iter_run = 0
        for it in range(self.max_iter):
            max_delta = 0.0
            for j in range(n_features):
                # ρ_j = X_jᵀ (residual + w_j X_j) = X_jᵀ residual + w_j ||X_j||²
                xj = Xc[:, j]
                rho = float(xj @ residual) + w[j] * col_norm_sq[j]
                w_new = _soft_threshold(rho, threshold) / col_norm_sq[j]
                delta = w_new - w[j]
                if delta != 0.0:
                    residual -= delta * xj
                    max_delta = max(max_delta, abs(delta))
                    w[j] = w_new
            n_iter_run = it + 1
            if max_delta < self.tol:
                break

        self.coef_ = w
        self.intercept_ = float(y_mean - X_mean @ w) if self.fit_intercept else 0.0
        self.n_iter_ = n_iter_run
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["coef_"])
        X = check_array(X)
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                f"Expected {self.coef_.shape[0]} features, got {X.shape[1]}."
            )
        return X @ self.coef_ + self.intercept_
