"""Linear Discriminant Analysis (closed-form), NumPy reference impl.

See ``PRINCIPLE.md`` for the derivation and ``EXTENSION.md`` for variants.
"""

from __future__ import annotations

import numpy as np

from algo_stack._base import BaseEstimator, ClassifierMixin
from algo_stack.utils.validation import check_array, check_X_y


class LinearDiscriminantAnalysis(BaseEstimator, ClassifierMixin):
    """Closed-form linear discriminant analysis for classification.

    Assumes class-conditional densities ``N(μ_k, Σ)`` sharing a pooled
    covariance. Decision boundaries are linear.

    Parameters
    ----------
    priors : ndarray of shape (n_classes,) or None, default None
        Class priors. If ``None``, estimated from training frequencies.
    shrinkage : float or None, default None
        Optional shrinkage toward the diagonal: ``(1 − γ) Σ + γ diag(Σ)``.
        ``None`` means no shrinkage.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
    means_ : ndarray of shape (n_classes, n_features)
    cov_ : ndarray of shape (n_features, n_features)
        Pooled covariance estimate.
    priors_ : ndarray of shape (n_classes,)
    coef_ : ndarray of shape (n_classes, n_features)
        Linear weights ``Σ⁻¹ μ_k`` for each class.
    intercept_ : ndarray of shape (n_classes,)
        ``−½ μ_kᵀ Σ⁻¹ μ_k + log π_k``.
    """

    def __init__(
        self,
        *,
        priors: np.ndarray | None = None,
        shrinkage: float | None = None,
    ) -> None:
        self.priors = priors
        self.shrinkage = shrinkage

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearDiscriminantAnalysis":
        X, y = check_X_y(X, y, y_numeric=False)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("LDA needs at least 2 classes in y.")

        n_samples, n_features = X.shape
        means = np.empty((n_classes, n_features), dtype=np.float64)
        counts = np.empty(n_classes, dtype=np.float64)

        # Pooled within-class scatter.
        cov = np.zeros((n_features, n_features), dtype=np.float64)
        for k in range(n_classes):
            Xk = X[y_idx == k]
            counts[k] = Xk.shape[0]
            means[k] = Xk.mean(axis=0)
            Xc = Xk - means[k]
            cov += Xc.T @ Xc

        # Unbiased pooled covariance (divide by n − K).
        dof = max(n_samples - n_classes, 1)
        cov /= dof

        if self.shrinkage is not None:
            gamma = float(self.shrinkage)
            if not 0.0 <= gamma <= 1.0:
                raise ValueError(f"shrinkage must be in [0, 1], got {gamma}.")
            cov = (1.0 - gamma) * cov + gamma * np.diag(np.diag(cov))

        # Regularise slightly if singular.
        cov = cov + 1e-6 * np.eye(n_features)

        if self.priors is None:
            priors = counts / n_samples
        else:
            priors = np.asarray(self.priors, dtype=np.float64)
            if priors.shape != (n_classes,):
                raise ValueError(
                    f"priors must have shape ({n_classes},), got {priors.shape}."
                )
            if not np.isclose(priors.sum(), 1.0):
                raise ValueError("priors must sum to 1.")

        # Solve Σ coefᵀ = meansᵀ  ⇒  coef = means @ Σ⁻¹  (via solve).
        # coef[k] = Σ⁻¹ μ_k
        coef = np.linalg.solve(cov, means.T).T  # (K, d)
        intercept = -0.5 * np.sum(means * coef, axis=1) + np.log(priors)

        self.means_ = means
        self.cov_ = cov
        self.priors_ = priors
        self.coef_ = coef
        self.intercept_ = intercept
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return linear discriminant scores, shape ``(n_samples, n_classes)``."""
        self._check_is_fitted(["coef_"])
        X = check_array(X)
        if X.shape[1] != self.coef_.shape[1]:
            raise ValueError(
                f"Expected {self.coef_.shape[1]} features, got {X.shape[1]}."
            )
        return X @ self.coef_.T + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_s = np.exp(scores)
        return exp_s / exp_s.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]
