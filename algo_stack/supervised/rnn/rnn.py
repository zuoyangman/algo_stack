"""Vanilla Recurrent Neural Network with Back-Propagation Through Time (BPTT)."""

from __future__ import annotations

import numpy as np

from algo_stack._base import BaseEstimator, ClassifierMixin
from algo_stack.utils import activations
from algo_stack.utils.validation import check_random_state


def _check_sequences(X: np.ndarray, name: str = "X") -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(
            f"Expected {name} with shape (n_samples, seq_len, n_features), "
            f"got ndim={arr.ndim}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf.")
    return arr


class RNNClassifier(BaseEstimator, ClassifierMixin):
    """Many-to-one vanilla RNN for sequence classification.

    Architecture: for each time step ``t``, update hidden state
    ``h_t = tanh(x_t W_xh + h_{t-1} W_hh + b_h)``; classify from the final
    ``h_T`` via a linear + softmax head.

    Parameters
    ----------
    hidden_size : int, default 16
    learning_rate : float, default 0.05
    n_epochs : int, default 100
    batch_size : int, default 32
    random_state : int | None, default None

    Attributes
    ----------
    classes_ : ndarray
    W_xh_, W_hh_, b_h_ : recurrent parameters
    W_hy_, b_y_ : output head
    loss_curve_, n_iter_
    """

    def __init__(
        self,
        *,
        hidden_size: int = 16,
        learning_rate: float = 0.05,
        n_epochs: int = 100,
        batch_size: int = 32,
        random_state: int | None = None,
    ) -> None:
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def _init_params(
        self, n_features: int, n_classes: int, rng: np.random.Generator
    ) -> None:
        h = self.hidden_size
        self.W_xh_ = rng.standard_normal((n_features, h)) * np.sqrt(1.0 / n_features)
        self.W_hh_ = rng.standard_normal((h, h)) * np.sqrt(1.0 / h)
        self.b_h_ = np.zeros(h)
        self.W_hy_ = rng.standard_normal((h, n_classes)) * np.sqrt(1.0 / h)
        self.b_y_ = np.zeros(n_classes)

    def _forward(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Return ``(probs, hidden_states, inputs_per_step)``."""

        N, T, _ = X.shape
        h = np.zeros((N, self.hidden_size))
        h_list: list[np.ndarray] = []
        x_list: list[np.ndarray] = []
        for t in range(T):
            x_list.append(X[:, t, :])
            z = x_list[-1] @ self.W_xh_ + h @ self.W_hh_ + self.b_h_
            h = activations.forward("tanh", z)
            h_list.append(h)
        logits = h @ self.W_hy_ + self.b_y_
        probs = activations.softmax(logits)
        return probs, h_list, x_list

    def _backward(
        self,
        probs: np.ndarray,
        Y: np.ndarray,
        h_list: list[np.ndarray],
        x_list: list[np.ndarray],
    ) -> dict[str, np.ndarray]:
        n = probs.shape[0]
        T = len(h_list)
        d_probs = (probs - Y) / n
        d_W_hy = h_list[-1].T @ d_probs
        d_b_y = d_probs.sum(axis=0)
        d_h = d_probs @ self.W_hy_.T

        d_W_xh = np.zeros_like(self.W_xh_)
        d_W_hh = np.zeros_like(self.W_hh_)
        d_b_h = np.zeros_like(self.b_h_)

        for t in reversed(range(T)):
            d_z = d_h * activations.grad("tanh", h_list[t])
            d_W_xh += x_list[t].T @ d_z
            d_b_h += d_z.sum(axis=0)
            if t > 0:
                d_W_hh += h_list[t - 1].T @ d_z
                d_h = d_z @ self.W_hh_.T
            else:
                d_h = np.zeros_like(d_h)

        return {
            "W_xh": d_W_xh,
            "W_hh": d_W_hh,
            "b_h": d_b_h,
            "W_hy": d_W_hy,
            "b_y": d_b_y,
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RNNClassifier":
        X = _check_sequences(X)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        K = len(self.classes_)
        if K < 2:
            raise ValueError("RNNClassifier requires at least 2 classes.")

        rng = check_random_state(self.random_state)
        self._init_params(X.shape[2], K, rng)
        Y_full = np.eye(K)[y_idx]
        n_samples = X.shape[0]
        self.loss_curve_ = []

        for epoch in range(self.n_epochs):
            indices = np.arange(n_samples)
            rng.shuffle(indices)
            for start in range(0, n_samples, self.batch_size):
                batch = indices[start : start + self.batch_size]
                Xb, Yb = X[batch], Y_full[batch]
                probs, h_list, x_list = self._forward(Xb)
                grads = self._backward(probs, Yb, h_list, x_list)
                lr = self.learning_rate
                self.W_xh_ -= lr * grads["W_xh"]
                self.W_hh_ -= lr * grads["W_hh"]
                self.b_h_ -= lr * grads["b_h"]
                self.W_hy_ -= lr * grads["W_hy"]
                self.b_y_ -= lr * grads["b_y"]

            probs_all, _, _ = self._forward(X)
            eps = 1e-12
            loss = float(-np.mean(np.sum(Y_full * np.log(probs_all + eps), axis=1)))
            self.loss_curve_.append(loss)
            self.n_iter_ = epoch + 1

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["W_xh_"])
        X = _check_sequences(X)
        probs, _, _ = self._forward(X)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
