"""Multi-layer perceptron (feed-forward NN) with full back-prop in NumPy.

Reference impl with:
  * arbitrary number of hidden layers (`hidden_layer_sizes` tuple)
  * activations: relu, tanh, sigmoid (hidden); identity (regression)
    or softmax (classification) on the output
  * SGD with momentum + L2 weight decay
  * mini-batch training with shuffling
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from algo_stack._base import BaseEstimator, ClassifierMixin, RegressorMixin
from algo_stack.utils.validation import check_array, check_random_state, check_X_y


_ACT_FORWARD = {
    "relu": lambda z: np.maximum(z, 0.0),
    "tanh": np.tanh,
    "sigmoid": lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500))),
}


def _act_grad(name: str, a: np.ndarray) -> np.ndarray:
    """Derivative of the activation evaluated on its *output* ``a``."""

    if name == "relu":
        return (a > 0.0).astype(a.dtype)
    if name == "tanh":
        return 1.0 - a * a
    if name == "sigmoid":
        return a * (1.0 - a)
    raise ValueError(f"Unknown activation {name!r}")


def _softmax(z: np.ndarray) -> np.ndarray:
    z_shift = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z_shift)
    return ez / ez.sum(axis=1, keepdims=True)


class _MLPBase(BaseEstimator):
    """Shared backbone: parameter init, forward pass, back-prop, training loop."""

    def __init__(
        self,
        *,
        hidden_layer_sizes: Sequence[int] = (32,),
        activation: str = "relu",
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        l2: float = 0.0,
        batch_size: int = 32,
        n_epochs: int = 200,
        tol: float = 1e-5,
        shuffle: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.activation = activation
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.l2 = l2
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.tol = tol
        self.shuffle = shuffle
        self.random_state = random_state

    # ---------- to be set by subclasses ----------
    _output_activation_: str  # "identity" or "softmax"
    _n_outputs_: int          # 1 for regression, K for classification

    def _validate_hyperparams(self) -> None:
        if self.activation not in _ACT_FORWARD:
            raise ValueError(
                f"activation must be one of {list(_ACT_FORWARD)}, got {self.activation!r}"
            )
        if any(h <= 0 for h in self.hidden_layer_sizes):
            raise ValueError("hidden_layer_sizes must contain positive integers.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must be in [0, 1).")

    def _init_params(self, n_features: int, rng: np.random.Generator) -> None:
        layer_sizes = (n_features, *self.hidden_layer_sizes, self._n_outputs_)
        self.coefs_: list[np.ndarray] = []
        self.intercepts_: list[np.ndarray] = []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # He initialisation for ReLU, otherwise Xavier/Glorot.
            if self.activation == "relu":
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(1.0 / fan_in)
            self.coefs_.append(rng.standard_normal((fan_in, fan_out)) * scale)
            self.intercepts_.append(np.zeros(fan_out))

        self._v_coefs_ = [np.zeros_like(W) for W in self.coefs_]
        self._v_intercepts_ = [np.zeros_like(b) for b in self.intercepts_]

    def _forward(self, X: np.ndarray) -> list[np.ndarray]:
        """Return the list of activations ``[a_0, a_1, …, a_L]``.

        ``a_0 = X``; for `l < L` apply the hidden activation; for ``l == L``
        apply the output activation (identity / softmax).
        """

        act = _ACT_FORWARD[self.activation]
        activations: list[np.ndarray] = [X]
        L = len(self.coefs_)
        for l, (W, b) in enumerate(zip(self.coefs_, self.intercepts_)):
            z = activations[-1] @ W + b
            if l < L - 1:
                a = act(z)
            else:
                if self._output_activation_ == "identity":
                    a = z
                elif self._output_activation_ == "softmax":
                    a = _softmax(z)
                else:
                    raise ValueError(self._output_activation_)
            activations.append(a)
        return activations

    def _backward(
        self,
        activations: list[np.ndarray],
        y_target: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        n = activations[0].shape[0]
        grads_W: list[np.ndarray] = [None] * len(self.coefs_)  # type: ignore[list-item]
        grads_b: list[np.ndarray] = [None] * len(self.coefs_)  # type: ignore[list-item]

        # Output-layer delta. For both (a) identity output + MSE loss, and
        # (b) softmax output + cross-entropy loss, the formula simplifies to
        # `delta = a_L − y_target` (with y_target being one-hot for case b).
        delta = (activations[-1] - y_target) / n

        for l in range(len(self.coefs_) - 1, -1, -1):
            grads_W[l] = activations[l].T @ delta
            grads_b[l] = delta.sum(axis=0)
            if self.l2:
                grads_W[l] = grads_W[l] + (self.l2 / n) * self.coefs_[l]
            if l > 0:
                delta = (delta @ self.coefs_[l].T) * _act_grad(
                    self.activation, activations[l]
                )
        return grads_W, grads_b

    def _update(self, grads_W: list[np.ndarray], grads_b: list[np.ndarray]) -> None:
        for l in range(len(self.coefs_)):
            self._v_coefs_[l] = (
                self.momentum * self._v_coefs_[l] - self.learning_rate * grads_W[l]
            )
            self._v_intercepts_[l] = (
                self.momentum * self._v_intercepts_[l]
                - self.learning_rate * grads_b[l]
            )
            self.coefs_[l] += self._v_coefs_[l]
            self.intercepts_[l] += self._v_intercepts_[l]

    def _epoch_loss(self, A_L: np.ndarray, y_target: np.ndarray) -> float:
        if self._output_activation_ == "identity":
            return float(np.mean((A_L - y_target) ** 2))
        # cross-entropy
        eps = 1e-12
        return float(-np.mean(np.sum(y_target * np.log(A_L + eps), axis=1)))

    def _fit_loop(self, X: np.ndarray, Y: np.ndarray) -> None:
        rng = check_random_state(self.random_state)
        n_samples = X.shape[0]
        self._init_params(X.shape[1], rng)
        self.loss_curve_: list[float] = []
        self.n_iter_ = 0
        prev_loss: float | None = None

        for epoch in range(self.n_epochs):
            indices = np.arange(n_samples)
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, n_samples, self.batch_size):
                batch = indices[start : start + self.batch_size]
                X_batch, Y_batch = X[batch], Y[batch]
                activations = self._forward(X_batch)
                grads_W, grads_b = self._backward(activations, Y_batch)
                self._update(grads_W, grads_b)

            # Track full-data loss at the end of each epoch.
            A_L = self._forward(X)[-1]
            loss = self._epoch_loss(A_L, Y)
            self.loss_curve_.append(loss)
            self.n_iter_ = epoch + 1
            if prev_loss is not None and abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss


class MLPRegressor(_MLPBase, RegressorMixin):
    """MLP for regression. Identity output + MSE loss.

    Attributes
    ----------
    coefs_       : list of weight matrices, one per layer.
    intercepts_  : list of bias vectors, one per layer.
    n_iter_      : epochs actually run.
    loss_curve_  : per-epoch MSE on the full training set.
    """

    _output_activation_ = "identity"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPRegressor":
        self._validate_hyperparams()
        X, y = check_X_y(X, y)
        self._n_outputs_ = 1
        Y = y.reshape(-1, 1).astype(np.float64)
        self._fit_loop(X, Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["coefs_"])
        X = check_array(X)
        return self._forward(X)[-1].ravel()


class MLPClassifier(_MLPBase, ClassifierMixin):
    """MLP for classification. Softmax output + cross-entropy loss.

    Attributes
    ----------
    classes_     : ndarray of unique class labels (sorted).
    coefs_       : list of weight matrices, one per layer.
    intercepts_  : list of bias vectors, one per layer.
    n_iter_      : epochs actually run.
    loss_curve_  : per-epoch cross-entropy on the full training set.
    """

    _output_activation_ = "softmax"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPClassifier":
        self._validate_hyperparams()
        X, y = check_X_y(X, y, y_numeric=False)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        K = len(self.classes_)
        if K < 2:
            raise ValueError("MLPClassifier requires at least 2 classes.")
        self._n_outputs_ = K
        Y = np.eye(K)[y_idx]
        self._fit_loop(X, Y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["coefs_"])
        X = check_array(X)
        return self._forward(X)[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
