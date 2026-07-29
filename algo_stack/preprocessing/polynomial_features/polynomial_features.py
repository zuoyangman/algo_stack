"""PolynomialFeatures: expand features into polynomial basis."""

from __future__ import annotations

from itertools import chain, combinations, combinations_with_replacement

import numpy as np

from algo_stack._base import BaseEstimator, TransformerMixin
from algo_stack.utils.validation import check_array


class PolynomialFeatures(BaseEstimator, TransformerMixin):
    """Generate a new feature matrix of polynomial combinations.

    For example, with ``degree=2`` and input features ``[a, b]`` the
    (non-interaction-only, with bias) output is::

        [1, a, b, a^2, a*b, b^2]

    Parameters
    ----------
    degree : int, default 2
        Maximum polynomial degree.
    include_bias : bool, default True
        If True, include a bias (constant-1) column.
    interaction_only : bool, default False
        If True, only products of *distinct* features are produced
        (no powers ``a^2``, ``a^3``, …).

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during ``fit``.
    n_output_features_ : int
        Number of output features after expansion.
    powers_ : ndarray of shape (n_output_features_, n_features_in_)
        Exponent matrix; row ``i`` is the multi-index of output feature ``i``.
    """

    def __init__(
        self,
        *,
        degree: int = 2,
        include_bias: bool = True,
        interaction_only: bool = False,
    ) -> None:
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> "PolynomialFeatures":
        X = check_array(X)
        if self.degree < 0:
            raise ValueError(f"degree must be >= 0, got {self.degree}.")

        n_features = X.shape[1]
        combos = self._combinations(n_features)
        powers = np.zeros((len(combos), n_features), dtype=np.int64)
        for i, combo in enumerate(combos):
            for feat_idx in combo:
                powers[i, feat_idx] += 1

        self.n_features_in_ = n_features
        self.n_output_features_ = len(combos)
        self.powers_ = powers
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["powers_", "n_features_in_"])
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but PolynomialFeatures is "
                f"expecting {self.n_features_in_} features."
            )

        n_samples = X.shape[0]
        XP = np.empty((n_samples, self.n_output_features_), dtype=np.float64)
        for i, exponents in enumerate(self.powers_):
            # Product over features of X[:, j] ** e_j  (empty product → 1).
            col = np.ones(n_samples, dtype=np.float64)
            for j, e in enumerate(exponents):
                if e == 0:
                    continue
                col *= X[:, j] ** int(e)
            XP[:, i] = col
        return XP

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def _combinations(self, n_features: int) -> list[tuple[int, ...]]:
        comb = combinations if self.interaction_only else combinations_with_replacement
        start = 0 if self.include_bias else 1
        return list(
            chain.from_iterable(
                comb(range(n_features), d) for d in range(start, self.degree + 1)
            )
        )
