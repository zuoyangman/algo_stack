"""Input validation helpers (numpy-array shape + dtype + finiteness)."""

from __future__ import annotations

from typing import Any

import numpy as np


def check_array(
    X: Any,
    *,
    ensure_2d: bool = True,
    dtype: Any = np.float64,
    ensure_finite: bool = True,
    name: str = "X",
) -> np.ndarray:
    """Cast ``X`` to a NumPy array and validate basic invariants.

    Parameters
    ----------
    X : array-like
        Input to validate.
    ensure_2d : bool, default True
        If True, raise unless ``X`` is 2-D. 1-D inputs are *not* silently
        promoted — that's a frequent source of bugs.
    dtype : numpy dtype, default ``np.float64``
        Cast to this dtype.
    ensure_finite : bool, default True
        Raise if ``X`` contains NaN or Inf.
    name : str
        Name of the variable for error messages.
    """

    arr = np.asarray(X, dtype=dtype)
    if ensure_2d and arr.ndim != 2:
        raise ValueError(
            f"Expected {name} to be 2-D (n_samples, n_features), got shape {arr.shape}."
        )
    if not ensure_2d and arr.ndim == 0:
        raise ValueError(f"Expected {name} to be at least 1-D, got scalar.")
    if ensure_finite and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf values.")
    return arr


def check_X_y(
    X: Any,
    y: Any,
    *,
    y_numeric: bool = True,
    multi_output: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate ``X`` and ``y`` jointly and return them as arrays."""

    X_arr = check_array(X, name="X")
    y_dtype = np.float64 if y_numeric else None
    y_arr = np.asarray(y, dtype=y_dtype)
    if not multi_output and y_arr.ndim != 1:
        raise ValueError(
            f"Expected y to be 1-D (n_samples,), got shape {y_arr.shape}."
        )
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"X and y have different numbers of samples: "
            f"{X_arr.shape[0]} vs {y_arr.shape[0]}."
        )
    if y_numeric and not np.all(np.isfinite(y_arr)):
        raise ValueError("y contains NaN or Inf values.")
    return X_arr, y_arr


def check_random_state(seed: Any) -> np.random.Generator:
    """Return a numpy ``Generator``; accept ``None``, ``int``, or ``Generator``."""

    if seed is None:
        return np.random.default_rng()
    if isinstance(seed, (int, np.integer)):
        return np.random.default_rng(int(seed))
    if isinstance(seed, np.random.Generator):
        return seed
    if isinstance(seed, np.random.RandomState):  # legacy support
        return np.random.default_rng(seed.randint(0, 2**31 - 1))
    raise TypeError(
        f"random_state must be None, int, or numpy Generator, got {type(seed)!r}."
    )
