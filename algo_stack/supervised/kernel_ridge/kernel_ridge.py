"""Kernel Ridge Regression — closed-form dual in RKHS (NumPy).

Solves ``α = (K + λ I)^{-1} y`` and predicts ``ŷ = K(x, X) α``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from algo_stack._base import BaseEstimator, RegressorMixin
from algo_stack.utils.kernels import kernel_matrix, resolve_gamma
from algo_stack.utils.validation import check_array, check_X_y


class KernelRidge(BaseEstimator, RegressorMixin):
    """Kernel Ridge Regression.

    Parameters
    ----------
    alpha : float, default 1.0
        Ridge strength ``λ`` on the dual: ``(K + α I) α_dual = y``.
    kernel : {"linear", "rbf", "polynomial"}, default "rbf"
    gamma : {"scale"} | float, default "scale"
        Kernel bandwidth / scale. ``"scale"`` → ``1 / (n_features * X.var())``.
    degree : int, default 3
        Degree for the polynomial kernel.
    coef0 : float, default 1.0
        Independent term for the polynomial kernel.
    fit_intercept : bool, default True
        If True, centre ``y`` before solving and store an intercept so that
        predictions are ``K(x, X) α + b``.

    Attributes
    ----------
    dual_coef_ : ndarray of shape (n_samples,)
    X_fit_ : ndarray of shape (n_samples, n_features)
    intercept_ : float
    gamma_ : float
        Resolved bandwidth used at fit time.
    """

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        kernel: str = "rbf",
        gamma: Any = "scale",
        degree: int = 3,
        coef0: float = 1.0,
        fit_intercept: bool = True,
    ) -> None:
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.fit_intercept = fit_intercept

    def _validate(self) -> None:
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0.")
        if self.kernel not in ("linear", "rbf", "polynomial", "poly"):
            raise ValueError(
                "kernel must be 'linear', 'rbf', or 'polynomial'."
            )
        if self.degree < 1:
            raise ValueError("degree must be >= 1.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KernelRidge":
        self._validate()
        X, y = check_X_y(X, y)
        gamma = resolve_gamma(self.gamma, X)
        self.gamma_ = gamma
        self.X_fit_ = X

        K = kernel_matrix(
            X,
            kernel=self.kernel,
            gamma=gamma,
            degree=self.degree,
            coef0=self.coef0,
        )
        if self.fit_intercept:
            y_c = y - y.mean()
        else:
            y_c = y

        n = K.shape[0]
        dual = np.linalg.solve(K + self.alpha * np.eye(n), y_c)
        self.dual_coef_ = dual

        if self.fit_intercept:
            pred = K @ dual
            self.intercept_ = float(y.mean() - pred.mean())
        else:
            self.intercept_ = 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["dual_coef_"])
        X = check_array(X)
        K = kernel_matrix(
            X,
            self.X_fit_,
            kernel=self.kernel,
            gamma=self.gamma_,
            degree=self.degree,
            coef0=self.coef0,
        )
        return K @ self.dual_coef_ + self.intercept_
