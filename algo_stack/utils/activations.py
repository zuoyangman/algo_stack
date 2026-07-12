"""Activation functions and their derivatives for neural-network modules.

Gradients are expressed as functions of the *activation output* ``a`` (not the
pre-activation ``z``), so forward passes only need to cache ``a``.
"""

from __future__ import annotations

import numpy as np

_ACTIVATIONS = ("relu", "tanh", "sigmoid", "identity", "step")


def forward(name: str, z: np.ndarray) -> np.ndarray:
    """Apply activation ``name`` to pre-activations ``z``."""

    if name == "relu":
        return np.maximum(z, 0.0)
    if name == "tanh":
        return np.tanh(z)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    if name == "identity":
        return z
    if name == "step":
        return (z >= 0.0).astype(z.dtype)
    raise ValueError(f"Unknown activation {name!r}; expected one of {_ACTIVATIONS}.")


def grad(name: str, a: np.ndarray) -> np.ndarray:
    """Derivative of activation w.r.t. pre-activation, evaluated at output ``a``."""

    if name == "relu":
        return (a > 0.0).astype(a.dtype)
    if name == "tanh":
        return 1.0 - a * a
    if name == "sigmoid":
        return a * (1.0 - a)
    if name == "identity":
        return np.ones_like(a)
    if name == "step":
        return np.zeros_like(a)
    raise ValueError(f"Unknown activation {name!r}; expected one of {_ACTIVATIONS}.")


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax (log-sum-exp stable)."""

    z_shift = z - z.max(axis=-1, keepdims=True)
    ez = np.exp(z_shift)
    return ez / ez.sum(axis=-1, keepdims=True)
