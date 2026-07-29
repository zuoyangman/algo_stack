"""CART decision trees (binary splits) — classifier (Gini) & regressor (MSE).

Didactic NumPy-only reference. Thresholds are midpoints between sorted unique
feature values (subsampled when there are many). Feature bagging via
``max_features`` is supported so Random Forests can reuse these trees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from algo_stack._base import BaseEstimator, ClassifierMixin, RegressorMixin
from algo_stack.utils.validation import check_array, check_X_y, check_random_state

# Cap on candidate thresholds per feature to keep fit cheap on dense columns.
_MAX_THRESHOLDS = 32


@dataclass
class Node:
    """Binary tree node; leaves have ``feature is None``."""

    feature: int | None
    threshold: float | None
    left: "Node | None"
    right: "Node | None"
    value: Any
    n_samples: int


def _gini(y: np.ndarray, n_classes: int) -> float:
    if y.size == 0:
        return 0.0
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    p = counts / y.size
    return float(1.0 - np.dot(p, p))


def _mse(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    mu = y.mean()
    return float(np.mean((y - mu) ** 2))


def _leaf_class_value(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Class probability vector (majority class recoverable via argmax)."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    if counts.sum() == 0:
        return np.zeros(n_classes)
    return counts / counts.sum()


def _leaf_reg_value(y: np.ndarray) -> float:
    return float(y.mean()) if y.size else 0.0


def _resolve_max_features(max_features: Any, n_features: int) -> int:
    if max_features is None:
        return n_features
    if isinstance(max_features, str):
        if max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if max_features == "log2":
            return max(1, int(np.log2(n_features)))
        raise ValueError(
            f"max_features={max_features!r} not supported; "
            "use None, int, float, 'sqrt', or 'log2'."
        )
    if isinstance(max_features, float):
        if not 0.0 < max_features <= 1.0:
            raise ValueError("float max_features must be in (0, 1].")
        return max(1, int(np.ceil(max_features * n_features)))
    max_features = int(max_features)
    if max_features < 1:
        raise ValueError("max_features must be >= 1.")
    return min(max_features, n_features)


def _candidate_thresholds(col: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    uniq = np.unique(col)
    if uniq.size < 2:
        return np.empty(0)
    mids = (uniq[:-1] + uniq[1:]) / 2.0
    if mids.size > _MAX_THRESHOLDS:
        idx = rng.choice(mids.size, size=_MAX_THRESHOLDS, replace=False)
        mids = np.sort(mids[idx])
    return mids


def _best_split_classification(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_classes: int,
    feature_indices: np.ndarray,
    min_samples_leaf: int,
    rng: np.random.Generator,
) -> tuple[int | None, float | None, float]:
    n = y.size
    parent = _gini(y, n_classes)
    # Start at -inf so zero-gain splits (e.g. first cut of XOR) are accepted;
    # deeper levels can still purify.
    best_gain = -np.inf
    best_feat: int | None = None
    best_thr: float | None = None

    for j in feature_indices:
        col = X[:, j]
        for thr in _candidate_thresholds(col, rng):
            left_mask = col <= thr
            n_left = int(left_mask.sum())
            n_right = n - n_left
            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue
            g = (
                parent
                - (n_left / n) * _gini(y[left_mask], n_classes)
                - (n_right / n) * _gini(y[~left_mask], n_classes)
            )
            if g > best_gain:
                best_gain = g
                best_feat = int(j)
                best_thr = float(thr)
    return best_feat, best_thr, best_gain


def _best_split_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_indices: np.ndarray,
    min_samples_leaf: int,
    rng: np.random.Generator,
) -> tuple[int | None, float | None, float]:
    n = y.size
    parent = _mse(y)
    best_gain = -np.inf
    best_feat: int | None = None
    best_thr: float | None = None

    for j in feature_indices:
        col = X[:, j]
        for thr in _candidate_thresholds(col, rng):
            left_mask = col <= thr
            n_left = int(left_mask.sum())
            n_right = n - n_left
            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue
            g = (
                parent
                - (n_left / n) * _mse(y[left_mask])
                - (n_right / n) * _mse(y[~left_mask])
            )
            if g > best_gain:
                best_gain = g
                best_feat = int(j)
                best_thr = float(thr)
    return best_feat, best_thr, best_gain


class _TreeBuilder:
    """Shared recursive grower for clf / reg trees."""

    def __init__(
        self,
        *,
        task: str,
        max_depth: int | None,
        min_samples_split: int,
        min_samples_leaf: int,
        max_features: Any,
        n_classes: int | None,
        rng: np.random.Generator,
    ) -> None:
        self.task = task
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_classes = n_classes
        self.rng = rng

    def build(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        n_samples, n_features = X.shape
        if self.task == "classification":
            assert self.n_classes is not None
            value: Any = _leaf_class_value(y, self.n_classes)
            pure = y.size > 0 and np.unique(y).size <= 1
        else:
            value = _leaf_reg_value(y)
            pure = False

        leaf = Node(
            feature=None,
            threshold=None,
            left=None,
            right=None,
            value=value,
            n_samples=n_samples,
        )

        if pure:
            return leaf
        if self.max_depth is not None and depth >= self.max_depth:
            return leaf
        if n_samples < self.min_samples_split:
            return leaf
        if n_samples < 2 * self.min_samples_leaf:
            return leaf

        k = _resolve_max_features(self.max_features, n_features)
        feat_idx = self.rng.choice(n_features, size=k, replace=False)

        if self.task == "classification":
            assert self.n_classes is not None
            feat, thr, gain = _best_split_classification(
                X,
                y,
                n_classes=self.n_classes,
                feature_indices=feat_idx,
                min_samples_leaf=self.min_samples_leaf,
                rng=self.rng,
            )
        else:
            feat, thr, gain = _best_split_regression(
                X,
                y,
                feature_indices=feat_idx,
                min_samples_leaf=self.min_samples_leaf,
                rng=self.rng,
            )

        # Accept any valid split (including zero-gain, needed for XOR-like data).
        if feat is None or thr is None or not np.isfinite(gain):
            return leaf

        left_mask = X[:, feat] <= thr
        left = self.build(X[left_mask], y[left_mask], depth + 1)
        right = self.build(X[~left_mask], y[~left_mask], depth + 1)
        return Node(
            feature=feat,
            threshold=thr,
            left=left,
            right=right,
            value=value,
            n_samples=n_samples,
        )


def _predict_tree_row(node: Node, x: np.ndarray) -> Any:
    while node.feature is not None:
        assert node.left is not None and node.right is not None
        if x[node.feature] <= node.threshold:  # type: ignore[operator]
            node = node.left
        else:
            node = node.right
    return node.value


class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """CART decision tree classifier (Gini impurity, binary splits).

    Parameters
    ----------
    max_depth : int or None, default None
        Maximum tree depth. ``None`` grows until purity / sample constraints.
    min_samples_split : int, default 2
        Minimum samples required to attempt a split.
    min_samples_leaf : int, default 1
        Minimum samples in each child of a split.
    max_features : int, float, {"sqrt","log2"} or None, default None
        Number of features considered at each split. ``None`` = all features.
    random_state : int | Generator | None, default None
        Controls feature / threshold subsampling randomness.
    """

    def __init__(
        self,
        *,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Any = None,
        random_state: int | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        X, y = check_X_y(X, y, y_numeric=False)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        rng = check_random_state(self.random_state)
        builder = _TreeBuilder(
            task="classification",
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            n_classes=self.n_classes_,
            rng=rng,
        )
        self.tree_ = builder.build(X, y_idx)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["tree_"])
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}."
            )
        proba = np.vstack([_predict_tree_row(self.tree_, row) for row in X])
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class DecisionTreeRegressor(BaseEstimator, RegressorMixin):
    """CART decision tree regressor (MSE impurity, binary splits).

    Parameters
    ----------
    max_depth : int or None, default None
    min_samples_split : int, default 2
    min_samples_leaf : int, default 1
    max_features : int, float, {"sqrt","log2"} or None, default None
    random_state : int | Generator | None, default None
    """

    def __init__(
        self,
        *,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Any = None,
        random_state: int | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressor":
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        rng = check_random_state(self.random_state)
        builder = _TreeBuilder(
            task="regression",
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            n_classes=None,
            rng=rng,
        )
        self.tree_ = builder.build(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["tree_"])
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}."
            )
        return np.array(
            [_predict_tree_row(self.tree_, row) for row in X], dtype=np.float64
        )
