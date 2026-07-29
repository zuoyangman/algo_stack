"""Classic (Rosenblatt) Perceptron with mistake-driven online learning."""

from __future__ import annotations

import numpy as np

from algo_stack._base import BaseEstimator, ClassifierMixin
from algo_stack.utils.validation import check_array, check_random_state, check_X_y


class Perceptron(BaseEstimator, ClassifierMixin):
    """Binary or multiclass Perceptron (one-vs-rest for K > 2).

    Uses the update rule ``w ← w + η · y · x`` on misclassified samples
    (with ``y ∈ {-1, +1}`` internally).

    Parameters
    ----------
    learning_rate : float, default 1.0
    n_iter : int, default 1000
        Maximum passes over the training set.
    tol : float, default 0.0
        Stop when the fraction of mistakes in an epoch falls below ``tol``.
    shuffle : bool, default True
    random_state : int | Generator | None, default None
    fit_intercept : bool, default True

    Attributes
    ----------
    classes_ : ndarray
    coef_ : ndarray of shape (n_classes, n_features) for multiclass, or
            (n_features,) for binary.
    intercept_ : ndarray of shape (n_classes,) or float for binary.
    n_iter_ : int
    mistakes_history_ : list[int]
        Number of mistakes per epoch.
    """

    def __init__(
        self,
        *,
        learning_rate: float = 1.0,
        n_iter: int = 1000,
        tol: float = 0.0,
        shuffle: bool = True,
        random_state: int | None = None,
        fit_intercept: bool = True,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.tol = tol
        self.shuffle = shuffle
        self.random_state = random_state
        self.fit_intercept = fit_intercept

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["coef_"])
        X = check_array(X)
        if self._binary_:
            return X @ self.coef_ + self.intercept_
        return X @ self.coef_.T + self.intercept_

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self._decision_function(X)
        if self._binary_:
            pred_idx = (scores >= 0.0).astype(int)
            return self.classes_[pred_idx]
        pred_idx = np.argmax(scores, axis=1)
        return self.classes_[pred_idx]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        X, y = check_X_y(X, y, y_numeric=False)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("Perceptron requires at least 2 classes.")

        self._binary_ = n_classes == 2
        n_samples, n_features = X.shape
        rng = check_random_state(self.random_state)

        if self._binary_:
            # Map to {-1, +1}
            y_signed = np.where(y_idx == 1, 1.0, -1.0)
            self.coef_ = np.zeros(n_features)
            self.intercept_ = 0.0
        else:
            # One-vs-rest: row k is the hyperplane for class k vs. rest.
            self.coef_ = np.zeros((n_classes, n_features))
            self.intercept_ = np.zeros(n_classes)
            y_signed = None  # type: ignore[assignment]

        mistakes_history: list[int] = []
        n_iter_run = 0

        for epoch in range(self.n_iter):
            indices = np.arange(n_samples)
            if self.shuffle:
                rng.shuffle(indices)
            mistakes = 0

            if self._binary_:
                for i in indices:
                    score = float(X[i] @ self.coef_ + self.intercept_)
                    pred = 1.0 if score >= 0.0 else -1.0
                    if pred != y_signed[i]:
                        self.coef_ += self.learning_rate * y_signed[i] * X[i]
                        if self.fit_intercept:
                            self.intercept_ += self.learning_rate * y_signed[i]
                        mistakes += 1
            else:
                for i in indices:
                    scores = X[i] @ self.coef_.T + self.intercept_
                    pred_k = int(np.argmax(scores))
                    true_k = int(y_idx[i])
                    if pred_k != true_k:
                        self.coef_[true_k] += self.learning_rate * X[i]
                        self.coef_[pred_k] -= self.learning_rate * X[i]
                        if self.fit_intercept:
                            self.intercept_[true_k] += self.learning_rate
                            self.intercept_[pred_k] -= self.learning_rate
                        mistakes += 1

            mistakes_history.append(mistakes)
            n_iter_run = epoch + 1
            if mistakes / n_samples <= self.tol:
                break

        self.n_iter_ = n_iter_run
        self.mistakes_history_ = mistakes_history
        return self
