"""Base classes and mixins shared by every algorithm in algo_stack.

The API intentionally mirrors a small subset of scikit-learn's estimator
protocol so that users coming from sklearn feel at home, while keeping the
implementation minimal and dependency-free.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from algo_stack.utils.metrics import accuracy_score, r2_score


class BaseEstimator:
    """Minimal estimator base class.

    Sub-classes must:

    * Accept all hyperparameters as **keyword-only** arguments in ``__init__``
      and simply store them on ``self`` (no work).
    * Implement ``fit(X, y=None)``; it must return ``self``.
    * Use trailing-underscore attributes for everything learned at fit time
      (e.g. ``self.coef_``).
    """

    def get_params(self) -> dict[str, Any]:
        """Return constructor hyperparameters as a dict.

        Convention: every public attribute *not* ending in ``_`` is a
        hyperparameter.
        """

        return {
            name: getattr(self, name)
            for name in vars(self)
            if not name.endswith("_") and not name.startswith("_")
        }

    def set_params(self, **params: Any) -> "BaseEstimator":
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(
                    f"{type(self).__name__!s} has no hyperparameter {k!r}"
                )
            setattr(self, k, v)
        return self

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({params})"

    def _check_is_fitted(self, attributes: list[str] | None = None) -> None:
        attributes = attributes or [a for a in vars(self) if a.endswith("_")]
        if not attributes or not any(hasattr(self, a) for a in attributes):
            raise RuntimeError(
                f"{type(self).__name__} instance is not fitted yet. "
                "Call `fit` before using this method."
            )


class RegressorMixin:
    """Mixin adding the default regression ``score`` (R²)."""

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)  # type: ignore[attr-defined]
        return r2_score(y, y_pred)


class ClassifierMixin:
    """Mixin adding the default classification ``score`` (accuracy)."""

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)  # type: ignore[attr-defined]
        return accuracy_score(y, y_pred)


class ClusterMixin:
    """Mixin adding ``fit_predict`` to clusterers."""

    def fit_predict(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        self.fit(X)  # type: ignore[attr-defined]
        return self.labels_  # type: ignore[attr-defined]


class TransformerMixin:
    """Mixin adding ``fit_transform`` to transformers."""

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, y).transform(X)  # type: ignore[attr-defined]
