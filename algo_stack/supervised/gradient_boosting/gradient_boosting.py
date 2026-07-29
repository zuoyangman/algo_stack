"""Gradient boosting — stagewise additive trees on residuals / gradients.

* ``GradientBoostingRegressor``: squared loss; init = mean(y); trees fit
  residuals.
* ``GradientBoostingClassifier``: binary logistic loss; init = log-odds of
  class prior; trees fit negative gradients; sigmoid for probabilities.
"""

from __future__ import annotations

import numpy as np

from algo_stack._base import BaseEstimator, ClassifierMixin, RegressorMixin
from algo_stack.supervised.decision_tree import DecisionTreeRegressor
from algo_stack.utils.validation import check_array, check_X_y, check_random_state


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


class GradientBoostingRegressor(BaseEstimator, RegressorMixin):
    """Gradient boosting regressor with squared loss.

    Parameters
    ----------
    n_estimators : int, default 100
        Number of boosting stages (trees).
    learning_rate : float, default 0.1
        Shrinkage applied to each tree's contribution.
    max_depth : int, default 3
        Depth of each regression tree.
    min_samples_split : int, default 2
    random_state : int | Generator | None, default None
    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingRegressor":
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        rng = check_random_state(self.random_state)

        self.init_ = float(y.mean())
        F = np.full(y.shape[0], self.init_, dtype=np.float64)
        self.estimators_: list[DecisionTreeRegressor] = []

        for _ in range(self.n_estimators):
            residual = y - F
            seed = int(rng.integers(0, 2**31 - 1))
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=seed,
            )
            tree.fit(X, residual)
            update = tree.predict(X)
            F = F + self.learning_rate * update
            self.estimators_.append(tree)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["estimators_", "init_"])
        X = check_array(X)
        F = np.full(X.shape[0], self.init_, dtype=np.float64)
        for tree in self.estimators_:
            F = F + self.learning_rate * tree.predict(X)
        return F


class GradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    """Binary gradient boosting classifier (logistic / binomial deviance).

    Initial prediction is the log-odds of the positive-class prior. Each tree
    is a regression tree fit to the negative gradient ``y − p``. Scores are
    mapped through a sigmoid for probabilities.

    Parameters
    ----------
    n_estimators : int, default 100
    learning_rate : float, default 0.1
    max_depth : int, default 3
    min_samples_split : int, default 2
    random_state : int | Generator | None, default None
    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingClassifier":
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        X, y = check_X_y(X, y, y_numeric=False)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError(
                "GradientBoostingClassifier currently supports binary y only; "
                f"got {len(self.classes_)} classes."
            )
        self.n_features_in_ = X.shape[1]
        y01 = y_idx.astype(np.float64)
        rng = check_random_state(self.random_state)

        # Log-odds of class-1 prior (with tiny smoothing).
        p = np.clip(y01.mean(), 1e-6, 1.0 - 1e-6)
        self.init_ = float(np.log(p / (1.0 - p)))
        F = np.full(y01.shape[0], self.init_, dtype=np.float64)
        self.estimators_: list[DecisionTreeRegressor] = []

        for _ in range(self.n_estimators):
            prob = _sigmoid(F)
            # Negative gradient of logistic loss w.r.t. F.
            residual = y01 - prob
            seed = int(rng.integers(0, 2**31 - 1))
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=seed,
            )
            tree.fit(X, residual)
            update = tree.predict(X)
            F = F + self.learning_rate * update
            self.estimators_.append(tree)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["estimators_", "init_"])
        X = check_array(X)
        F = np.full(X.shape[0], self.init_, dtype=np.float64)
        for tree in self.estimators_:
            F = F + self.learning_rate * tree.predict(X)
        return F

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        p1 = _sigmoid(scores)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
