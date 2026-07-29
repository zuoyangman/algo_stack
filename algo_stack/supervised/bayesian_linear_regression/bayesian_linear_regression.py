"""Bayesian linear regression with an isotropic Gaussian prior.

See ``PRINCIPLE.md`` for the derivation and ``EXTENSION.md`` for variants.
"""

from __future__ import annotations

import numpy as np

from algo_stack._base import BaseEstimator, RegressorMixin
from algo_stack.utils.validation import check_array, check_X_y


class BayesianLinearRegression(BaseEstimator, RegressorMixin):
    """Bayesian linear regression with isotropic Gaussian weight prior.

    Prior:       ``w ~ N(0, α⁻¹ I)``
    Likelihood:  ``y | X, w ~ N(X w (+ b), β⁻¹ I)``

    The posterior is Gaussian with closed-form mean / covariance. ``predict``
    returns the predictive mean; ``predict_std`` returns the predictive
    standard deviation.

    Parameters
    ----------
    alpha : float, default 1.0
        Prior precision (``α``). Larger → stronger shrinkage toward 0.
    beta : float, default 1.0
        Noise precision (``β``). Larger → less observation noise.
    fit_intercept : bool, default True
        Whether to estimate an intercept (unregularised empirical mean).

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Posterior mean of the weights.
    intercept_ : float
    alpha_ : float
        Prior precision used at fit time.
    beta_ : float
        Noise precision used at fit time.
    sigma_ : ndarray of shape (n_features, n_features)
        Posterior covariance of the weights (centred design).
    """

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
        fit_intercept: bool = True,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.fit_intercept = fit_intercept

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianLinearRegression":
        if self.alpha <= 0:
            raise ValueError(f"alpha (prior precision) must be > 0, got {self.alpha}.")
        if self.beta <= 0:
            raise ValueError(f"beta (noise precision) must be > 0, got {self.beta}.")
        X, y = check_X_y(X, y)
        n_features = X.shape[1]

        if self.fit_intercept:
            X_mean = X.mean(axis=0)
            y_mean = float(y.mean())
            Xc = X - X_mean
            yc = y - y_mean
        else:
            X_mean = np.zeros(n_features)
            y_mean = 0.0
            Xc, yc = X, y

        # Posterior precision: S_N⁻¹ = α I + β Xᵀ X
        # Posterior mean:      m_N  = β S_N Xᵀ y
        XtX = Xc.T @ Xc
        A = self.beta * XtX
        A.flat[:: n_features + 1] += self.alpha
        # S_N = A⁻¹; m_N = β A⁻¹ Xᵀ y
        sigma = np.linalg.inv(A)
        coef = self.beta * (sigma @ (Xc.T @ yc))

        self.coef_ = coef
        self.intercept_ = float(y_mean - X_mean @ coef) if self.fit_intercept else 0.0
        self.alpha_ = float(self.alpha)
        self.beta_ = float(self.beta)
        self.sigma_ = sigma
        self._X_mean_ = X_mean
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the predictive mean ``E[y* | X*]``."""
        self._check_is_fitted(["coef_"])
        X = check_array(X)
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                f"Expected {self.coef_.shape[0]} features, got {X.shape[1]}."
            )
        return X @ self.coef_ + self.intercept_

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """Return the predictive standard deviation for each sample.

        ``σ²(x*) = β⁻¹ + (x* − x̄)ᵀ S_N (x* − x̄)``
        """
        self._check_is_fitted(["sigma_"])
        X = check_array(X)
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                f"Expected {self.coef_.shape[0]} features, got {X.shape[1]}."
            )
        Xc = X - self._X_mean_
        # diag(Xc S Xcᵀ) = sum((Xc @ S) * Xc, axis=1)
        quad = np.sum((Xc @ self.sigma_) * Xc, axis=1)
        var = 1.0 / self.beta_ + quad
        return np.sqrt(var)

    def predict_var(self, X: np.ndarray) -> np.ndarray:
        """Return the predictive variance for each sample."""
        std = self.predict_std(X)
        return std * std
