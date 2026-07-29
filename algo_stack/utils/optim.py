"""Gradient-based optimisers shared by neural-network modules."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Optimizer(Protocol):
    """Minimal optimiser protocol: reset once, then step many times."""

    def reset(self, params: list[np.ndarray]) -> None: ...
    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None: ...


class SGDOptimizer:
    """Plain stochastic / mini-batch gradient descent."""

    def __init__(self, *, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate

    def reset(self, params: list[np.ndarray]) -> None:
        return None

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        for p, g in zip(params, grads):
            p -= self.learning_rate * g


class SGDMomentumOptimizer:
    """SGD with classical momentum."""

    def __init__(self, *, learning_rate: float = 0.01, momentum: float = 0.9) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
        self._velocity: list[np.ndarray] | None = None

    def reset(self, params: list[np.ndarray]) -> None:
        self._velocity = [np.zeros_like(p) for p in params]

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        if self._velocity is None:
            self.reset(params)
        assert self._velocity is not None
        for i, (p, g) in enumerate(zip(params, grads)):
            self._velocity[i] = self.momentum * self._velocity[i] - self.learning_rate * g
            p += self._velocity[i]


class AdamOptimizer:
    """Adam optimiser (Kingma & Ba 2015)."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m: list[np.ndarray] | None = None
        self._v: list[np.ndarray] | None = None
        self._t = 0

    def reset(self, params: list[np.ndarray]) -> None:
        self._m = [np.zeros_like(p) for p in params]
        self._v = [np.zeros_like(p) for p in params]
        self._t = 0

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        if self._m is None or self._v is None:
            self.reset(params)
        assert self._m is not None and self._v is not None
        self._t += 1
        bc1 = 1.0 - self.beta1**self._t
        bc2 = 1.0 - self.beta2**self._t
        for i, (p, g) in enumerate(zip(params, grads)):
            self._m[i] = self.beta1 * self._m[i] + (1.0 - self.beta1) * g
            self._v[i] = self.beta2 * self._v[i] + (1.0 - self.beta2) * (g * g)
            m_hat = self._m[i] / bc1
            v_hat = self._v[i] / bc2
            p -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)


def make_optimizer(
    name: str,
    *,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> Optimizer:
    """Factory: ``"sgd"`` | ``"momentum"`` | ``"adam"``."""

    if name == "sgd":
        return SGDOptimizer(learning_rate=learning_rate)
    if name == "momentum":
        return SGDMomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
    if name == "adam":
        return AdamOptimizer(
            learning_rate=learning_rate, beta1=beta1, beta2=beta2, eps=eps
        )
    raise ValueError(f"Unknown optimizer {name!r}; expected 'sgd', 'momentum', or 'adam'.")
