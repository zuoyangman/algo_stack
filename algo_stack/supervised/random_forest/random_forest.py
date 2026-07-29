"""Random Forests — bagged decision trees with feature randomness.

Reuses ``DecisionTreeClassifier`` / ``DecisionTreeRegressor``. Each tree is
fit on a bootstrap sample; at every split the tree considers only
``max_features`` features (default ``"sqrt"``). Predictions average votes
(classifier) or leaf means (regressor).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from algo_stack._base import BaseEstimator, ClassifierMixin, RegressorMixin
from algo_stack.supervised.decision_tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)
from algo_stack.utils.validation import check_array, check_X_y, check_random_state


class RandomForestClassifier(BaseEstimator, ClassifierMixin):
    """Bootstrap-aggregated decision tree classifier.

    Parameters
    ----------
    n_estimators : int, default 100
        Number of trees.
    max_depth : int or None, default None
        Passed to each ``DecisionTreeClassifier``.
    min_samples_split : int, default 2
    max_features : int, float, {"sqrt","log2"} or None, default "sqrt"
        Features considered per split inside each tree.
    random_state : int | Generator | None, default None
    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        max_features: Any = "sqrt",
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")
        X, y = check_X_y(X, y, y_numeric=False)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]
        rng = check_random_state(self.random_state)

        self.estimators_: list[DecisionTreeClassifier] = []
        for _ in range(self.n_estimators):
            seed = int(rng.integers(0, 2**31 - 1))
            boot = rng.integers(0, n_samples, size=n_samples)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=seed,
            )
            # Fit with original labels so classes_ stay aligned.
            tree.fit(X[boot], y[boot])
            self.estimators_.append(tree)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["estimators_"])
        X = check_array(X)
        # Average soft votes; map each tree's class order onto forest classes_.
        n = X.shape[0]
        n_classes = len(self.classes_)
        acc = np.zeros((n, n_classes), dtype=np.float64)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        for tree in self.estimators_:
            p = tree.predict_proba(X)
            for j, c in enumerate(tree.classes_):
                acc[:, class_to_idx[c]] += p[:, j]
        acc /= len(self.estimators_)
        return acc

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class RandomForestRegressor(BaseEstimator, RegressorMixin):
    """Bootstrap-aggregated decision tree regressor.

    Parameters
    ----------
    n_estimators : int, default 100
    max_depth : int or None, default None
    min_samples_split : int, default 2
    max_features : int, float, {"sqrt","log2"} or None, default "sqrt"
    random_state : int | Generator | None, default None
    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        max_features: Any = "sqrt",
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressor":
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]
        rng = check_random_state(self.random_state)

        self.estimators_: list[DecisionTreeRegressor] = []
        for _ in range(self.n_estimators):
            seed = int(rng.integers(0, 2**31 - 1))
            boot = rng.integers(0, n_samples, size=n_samples)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=seed,
            )
            tree.fit(X[boot], y[boot])
            self.estimators_.append(tree)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["estimators_"])
        X = check_array(X)
        preds = np.column_stack([t.predict(X) for t in self.estimators_])
        return preds.mean(axis=1)
