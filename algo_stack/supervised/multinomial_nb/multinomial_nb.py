"""Multinomial Naive Bayes for non-negative count features.

See ``PRINCIPLE.md`` for the derivation and ``EXTENSION.md`` for variants.
"""

from __future__ import annotations

import numpy as np

from algo_stack._base import BaseEstimator, ClassifierMixin
from algo_stack.utils.validation import check_array, check_X_y


class MultinomialNB(BaseEstimator, ClassifierMixin):
    """Multinomial Naive Bayes classifier for count / TF feature vectors.

    Models ``p(x | y = k)`` as a multinomial distribution over feature counts
    with Laplace (add-``alpha``) smoothing. All prediction arithmetic is done
    in log-space for numerical stability.

    Parameters
    ----------
    alpha : float, default 1.0
        Additive (Laplace/Lidstone) smoothing parameter. Must be ≥ 0.
    fit_prior : bool, default True
        Whether to learn class priors from data. If ``False``, a uniform prior
        is used.
    class_prior : ndarray of shape (n_classes,) or None, default None
        Explicit priors; overrides ``fit_prior`` when given.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
    class_count_ : ndarray of shape (n_classes,)
    class_log_prior_ : ndarray of shape (n_classes,)
    feature_count_ : ndarray of shape (n_classes, n_features)
    feature_log_prob_ : ndarray of shape (n_classes, n_features)
        ``log P(feature_j | class_k)``.
    """

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        fit_prior: bool = True,
        class_prior: np.ndarray | None = None,
    ) -> None:
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultinomialNB":
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}.")
        X, y = check_X_y(X, y, y_numeric=False)
        if np.any(X < 0):
            raise ValueError("MultinomialNB requires non-negative features.")

        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("MultinomialNB needs at least 2 classes in y.")

        n_samples, n_features = X.shape
        feature_count = np.zeros((n_classes, n_features), dtype=np.float64)
        class_count = np.zeros(n_classes, dtype=np.float64)

        for k in range(n_classes):
            mask = y_idx == k
            class_count[k] = mask.sum()
            feature_count[k] = X[mask].sum(axis=0)

        # Laplace-smoothed feature log-probabilities.
        smoothed = feature_count + self.alpha
        smoothed_sum = smoothed.sum(axis=1, keepdims=True)
        feature_log_prob = np.log(smoothed) - np.log(smoothed_sum)

        if self.class_prior is not None:
            priors = np.asarray(self.class_prior, dtype=np.float64)
            if priors.shape != (n_classes,):
                raise ValueError(
                    f"class_prior must have shape ({n_classes},), got {priors.shape}."
                )
            class_log_prior = np.log(priors)
        elif self.fit_prior:
            class_log_prior = np.log(class_count) - np.log(class_count.sum())
        else:
            class_log_prior = np.full(n_classes, -np.log(n_classes))

        self.class_count_ = class_count
        self.feature_count_ = feature_count
        self.feature_log_prob_ = feature_log_prob
        self.class_log_prior_ = class_log_prior
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        # log P(x, y=k) = log π_k + Σ_j x_j log θ_{k,j}
        return X @ self.feature_log_prob_.T + self.class_log_prior_

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["feature_log_prob_"])
        X = check_array(X)
        if np.any(X < 0):
            raise ValueError("MultinomialNB requires non-negative features.")
        if X.shape[1] != self.feature_log_prob_.shape[1]:
            raise ValueError(
                f"Expected {self.feature_log_prob_.shape[1]} features, "
                f"got {X.shape[1]}."
            )
        jll = self._joint_log_likelihood(X)
        # log-sum-exp normalisation
        log_prob_x = np.logaddexp.reduce(jll, axis=1, keepdims=True)
        return jll - log_prob_x

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["feature_log_prob_"])
        X = check_array(X)
        if np.any(X < 0):
            raise ValueError("MultinomialNB requires non-negative features.")
        if X.shape[1] != self.feature_log_prob_.shape[1]:
            raise ValueError(
                f"Expected {self.feature_log_prob_.shape[1]} features, "
                f"got {X.shape[1]}."
            )
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]
