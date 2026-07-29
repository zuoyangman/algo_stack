"""Ridge regression (OLS + L2), closed-form NumPy reference impl.

See ``PRINCIPLE.md`` for the derivation and ``EXTENSION.md`` for variants.
"""

from __future__ import annotations

import numpy as np

from algo_stack._base import BaseEstimator, RegressorMixin
from algo_stack.utils.validation import check_array, check_X_y


class Ridge(BaseEstimator, RegressorMixin):
    """Ridge regression: ordinary least squares with L2 weight penalty.

    Solves

        min_{w,b}  ||X w + b − y||²  +  α ||w||²

    in closed form via the normal equations.  The intercept is **not**
    penalised.

    Parameters
    ----------
    alpha : float, default 1.0
        L2 regularisation strength. Must be ≥ 0.
    fit_intercept : bool, default True
        Whether to estimate an intercept term.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
    intercept_ : float
    """

    def __init__(self, *, alpha: float = 1.0, fit_intercept: bool = True) -> None:
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Ridge":
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}.")
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

        # (XᵀX + α I) w = Xᵀ y   — intercept column is already removed by centring
        XtX = Xc.T @ Xc
        XtX.flat[:: n_features + 1] += self.alpha  # add α to diagonal
        w = np.linalg.solve(XtX, Xc.T @ yc)

        self.coef_ = w
        self.intercept_ = float(y_mean - X_mean @ w) if self.fit_intercept else 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["coef_"])
        X = check_array(X)
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                f"Expected {self.coef_.shape[0]} features, got {X.shape[1]}."
            )
        return X @ self.coef_ + self.intercept_
