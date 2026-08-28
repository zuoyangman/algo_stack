"""Ordinary Least Squares (OLS) linear regression, NumPy reference impl.

See ``PRINCIPLE.md`` for the derivation and ``EXTENSION.md`` for variants
(ridge, lasso, GLS, weighted least squares, …).
"""

from __future__ import annotations

import numpy as np

from algo_stack._base import BaseEstimator, RegressorMixin
from algo_stack.utils.validation import check_array, check_X_y


class LinearRegression(BaseEstimator, RegressorMixin):
    """Ordinary-least-squares linear regression.

    Solves ``min_w ||X w + b - y||²`` either via the normal equations
    (``solver="normal"``) or via ``numpy.linalg.lstsq`` (``solver="lstsq"``,
    default — numerically safer).

    Parameters
    ----------
    fit_intercept : bool, default True
        Whether to estimate an intercept term.
    solver : {"lstsq", "normal"}, default "lstsq"
        Which linear-system solver to use.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
    intercept_ : float
    rank_ : int
        Rank of the design matrix (only when ``solver="lstsq"``).
    """

    def __init__(self, *, fit_intercept: bool = True, solver: str = "lstsq") -> None:
        self.fit_intercept = fit_intercept
        self.solver = solver

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        X, y = check_X_y(X, y)

        if self.fit_intercept:
            # Augment X with a column of ones so the intercept becomes the last weight.
            X_design = np.hstack([X, np.ones((X.shape[0], 1))])
        else:
            X_design = X

        if self.solver == "lstsq":
            theta, _residuals, rank, _sv = np.linalg.lstsq(X_design, y, rcond=None)
            self.rank_ = int(rank)
        elif self.solver == "normal":
            # Normal equations: (XᵀX) θ = Xᵀ y. Use solve (faster, more stable
            # than computing an explicit inverse).
            XtX = X_design.T @ X_design
            Xty = X_design.T @ y
            theta = np.linalg.solve(XtX, Xty)
            self.rank_ = int(np.linalg.matrix_rank(X_design))
        else:
            raise ValueError(
                f"Unknown solver {self.solver!r}; expected 'lstsq' or 'normal'."
            )

        if self.fit_intercept:
            self.coef_ = theta[:-1]
            self.intercept_ = float(theta[-1])
        else:
            self.coef_ = theta
            self.intercept_ = 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["coef_"])
        X = check_array(X)
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                f"Expected {self.coef_.shape[0]} features, got {X.shape[1]}."
            )
        return X @ self.coef_ + self.intercept_
