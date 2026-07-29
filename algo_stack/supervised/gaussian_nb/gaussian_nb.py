"""Gaussian Naive Bayes classifier, NumPy reference impl.

See ``PRINCIPLE.md`` for the derivation and ``EXTENSION.md`` for variants.
"""

from __future__ import annotations

import numpy as np

from algo_stack._base import BaseEstimator, ClassifierMixin
from algo_stack.utils.validation import check_array, check_X_y


class GaussianNB(BaseEstimator, ClassifierMixin):
    """Gaussian Naive Bayes classifier.

    Assumes features are conditionally independent given the class and each
    follows a univariate Gaussian ``N(θ_{k,j}, σ²_{k,j})``.

    Parameters
    ----------
    var_smoothing : float, default 1e-9
        Portion of the largest feature variance added to all variances for
        numerical stability (mirrors sklearn's ``var_smoothing``).
    priors : ndarray of shape (n_classes,) or None, default None
        Class priors. If ``None``, estimated from training frequencies.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
    class_prior_ : ndarray of shape (n_classes,)
    theta_ : ndarray of shape (n_classes, n_features)
        Per-class feature means.
    var_ : ndarray of shape (n_classes, n_features)
        Per-class feature variances.
    """

    def __init__(
        self,
        *,
        var_smoothing: float = 1e-9,
        priors: np.ndarray | None = None,
    ) -> None:
        self.var_smoothing = var_smoothing
        self.priors = priors

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNB":
        X, y = check_X_y(X, y, y_numeric=False)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("GaussianNB needs at least 2 classes in y.")

        n_samples, n_features = X.shape
        theta = np.empty((n_classes, n_features), dtype=np.float64)
        var = np.empty((n_classes, n_features), dtype=np.float64)
        counts = np.empty(n_classes, dtype=np.float64)

        for k in range(n_classes):
            Xk = X[y_idx == k]
            counts[k] = Xk.shape[0]
            theta[k] = Xk.mean(axis=0)
            # Population variance (ddof=0), matching common NB implementations.
            var[k] = Xk.var(axis=0)

        # Floor variances for stability.
        epsilon = self.var_smoothing * max(float(X.var(axis=0).max()), 1e-12)
        var = var + epsilon

        if self.priors is None:
            class_prior = counts / n_samples
        else:
            class_prior = np.asarray(self.priors, dtype=np.float64)
            if class_prior.shape != (n_classes,):
                raise ValueError(
                    f"priors must have shape ({n_classes},), got {class_prior.shape}."
                )

        self.theta_ = theta
        self.var_ = var
        self.class_prior_ = class_prior
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Return log P(x, y=k) for each sample/class, shape (n, K)."""
        n_classes, n_features = self.theta_.shape
        log_prior = np.log(self.class_prior_)
        # log N(x; μ, σ²) = −½ log(2πσ²) − (x−μ)² / (2σ²)
        ll = np.empty((X.shape[0], n_classes), dtype=np.float64)
        for k in range(n_classes):
            # Broadcast over features, sum independence assumption.
            log_prob = -0.5 * np.log(2.0 * np.pi * self.var_[k])
            log_prob = log_prob - 0.5 * (X - self.theta_[k]) ** 2 / self.var_[k]
            ll[:, k] = log_prob.sum(axis=1) + log_prior[k]
        return ll

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["theta_"])
        X = check_array(X)
        if X.shape[1] != self.theta_.shape[1]:
            raise ValueError(
                f"Expected {self.theta_.shape[1]} features, got {X.shape[1]}."
            )
        jll = self._joint_log_likelihood(X)
        jll = jll - jll.max(axis=1, keepdims=True)
        exp_jll = np.exp(jll)
        return exp_jll / exp_jll.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["theta_"])
        X = check_array(X)
        if X.shape[1] != self.theta_.shape[1]:
            raise ValueError(
                f"Expected {self.theta_.shape[1]} features, got {X.shape[1]}."
            )
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]
