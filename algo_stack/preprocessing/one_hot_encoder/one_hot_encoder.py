"""OneHotEncoder: encode categorical columns as dense indicator columns."""

from __future__ import annotations

from typing import Any

import numpy as np

from algo_stack._base import BaseEstimator, TransformerMixin
from algo_stack.utils.validation import check_array


class OneHotEncoder(BaseEstimator, TransformerMixin):
    """One-hot (dummy) encode categorical features.

    Accepts a 1-D array (single feature) or a 2-D array (one categorical
    column per feature).  Unknown categories at ``transform`` time produce
    an all-zero row for that feature block.

    Parameters
    ----------
    drop : {None, 'first'}, default None
        If ``'first'``, drop the first category of each feature (useful to
        avoid the dummy-variable trap in linear models).  ``None`` keeps
        every category.

    Attributes
    ----------
    categories_ : list of ndarray
        The categories of each feature determined during ``fit``, in the
        order of the corresponding output columns.
    """

    def __init__(self, *, drop: str | None = None) -> None:
        self.drop = drop

    def fit(self, X: Any, y: np.ndarray | None = None) -> "OneHotEncoder":
        if self.drop not in (None, "first"):
            raise ValueError(
                f"drop must be None or 'first', got {self.drop!r}."
            )
        X = self._validate_input(X)
        n_features = X.shape[1]
        categories: list[np.ndarray] = []
        for j in range(n_features):
            # Preserve order of first appearance while uniquifying.
            col = X[:, j]
            _, idx = np.unique(col, return_index=True)
            cats = col[np.sort(idx)]
            if self.drop == "first":
                if cats.size == 0:
                    raise ValueError(f"Feature {j} has no categories.")
                cats = cats[1:]
            categories.append(cats)
        self.categories_ = categories
        return self

    def transform(self, X: Any) -> np.ndarray:
        self._check_is_fitted(["categories_"])
        X = self._validate_input(X)
        if X.shape[1] != len(self.categories_):
            raise ValueError(
                f"X has {X.shape[1]} features, but OneHotEncoder is "
                f"expecting {len(self.categories_)} features."
            )

        n_samples = X.shape[0]
        blocks: list[np.ndarray] = []
        for j, cats in enumerate(self.categories_):
            n_cats = cats.size
            block = np.zeros((n_samples, n_cats), dtype=np.float64)
            col = X[:, j]
            for k, cat in enumerate(cats):
                block[col == cat, k] = 1.0
            blocks.append(block)

        if not blocks:
            return np.zeros((n_samples, 0), dtype=np.float64)
        return np.hstack(blocks)

    def fit_transform(self, X: Any, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    @staticmethod
    def _validate_input(X: Any) -> np.ndarray:
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim != 2:
            raise ValueError(
                f"Expected X to be 1-D or 2-D, got shape {arr.shape}."
            )
        if arr.shape[0] == 0:
            raise ValueError("X has zero samples.")
        # Re-run through check_array for finite numeric inputs; object /
        # string categoricals skip the float cast.
        if arr.dtype.kind in "biufc":
            return check_array(arr, dtype=np.float64)
        return arr
