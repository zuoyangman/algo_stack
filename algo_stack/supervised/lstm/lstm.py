"""LSTM and GRU sequence classifiers with Back-Propagation Through Time."""

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


def _clip_grads(grads: dict[str, np.ndarray], max_norm: float = 5.0) -> None:
    total = float(np.sqrt(sum(np.sum(g * g) for g in grads.values())))
    if total > max_norm:
        scale = max_norm / total
        for k in grads:
            grads[k] *= scale


class LSTMClassifier(BaseEstimator, ClassifierMixin):
    """Many-to-one LSTM for sequence classification.

    Standard LSTM cell with input / forget / output / candidate gates.
    Classification uses the final hidden state ``h_T`` via a linear + softmax head.

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
    W_x_, W_h_, b_ : LSTM gate parameters (concatenated i,f,o,g)
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
        self.W_x_ = rng.standard_normal((n_features, 4 * h)) * np.sqrt(1.0 / n_features)
        self.W_h_ = rng.standard_normal((h, 4 * h)) * np.sqrt(1.0 / h)
        self.b_ = np.zeros(4 * h)
        # Forget-gate bias init encourages remembering early on.
        self.b_[h : 2 * h] = 1.0
        self.W_hy_ = rng.standard_normal((h, n_classes)) * np.sqrt(1.0 / h)
        self.b_y_ = np.zeros(n_classes)

    def _forward(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
        N, T, _ = X.shape
        h_dim = self.hidden_size
        h = np.zeros((N, h_dim))
        c = np.zeros((N, h_dim))
        cache: list[dict[str, np.ndarray]] = []

        for t in range(T):
            x_t = X[:, t, :]
            h_prev, c_prev = h, c
            pre = x_t @ self.W_x_ + h_prev @ self.W_h_ + self.b_
            i = activations.forward("sigmoid", pre[:, :h_dim])
            f = activations.forward("sigmoid", pre[:, h_dim : 2 * h_dim])
            o = activations.forward("sigmoid", pre[:, 2 * h_dim : 3 * h_dim])
            g = activations.forward("tanh", pre[:, 3 * h_dim :])
            c = f * c_prev + i * g
            tanh_c = np.tanh(c)
            h = o * tanh_c
            cache.append(
                {
                    "x": x_t,
                    "h_prev": h_prev,
                    "c_prev": c_prev,
                    "i": i,
                    "f": f,
                    "o": o,
                    "g": g,
                    "c": c,
                    "tanh_c": tanh_c,
                    "h": h,
                }
            )

        logits = h @ self.W_hy_ + self.b_y_
        probs = activations.softmax(logits)
        return probs, cache

    def _backward(
        self,
        probs: np.ndarray,
        Y: np.ndarray,
        cache: list[dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        n = probs.shape[0]
        h_dim = self.hidden_size
        d_probs = (probs - Y) / n
        d_W_hy = cache[-1]["h"].T @ d_probs
        d_b_y = d_probs.sum(axis=0)
        d_h = d_probs @ self.W_hy_.T
        d_c = np.zeros_like(d_h)

        d_W_x = np.zeros_like(self.W_x_)
        d_W_h = np.zeros_like(self.W_h_)
        d_b = np.zeros_like(self.b_)

        for t in reversed(range(len(cache))):
            step = cache[t]
            d_o = d_h * step["tanh_c"]
            d_tanh_c = d_h * step["o"]
            d_c = d_c + d_tanh_c * (1.0 - step["tanh_c"] ** 2)

            d_i = d_c * step["g"]
            d_f = d_c * step["c_prev"]
            d_g = d_c * step["i"]
            d_c_prev = d_c * step["f"]

            d_i_pre = d_i * activations.grad("sigmoid", step["i"])
            d_f_pre = d_f * activations.grad("sigmoid", step["f"])
            d_o_pre = d_o * activations.grad("sigmoid", step["o"])
            d_g_pre = d_g * activations.grad("tanh", step["g"])
            d_pre = np.concatenate([d_i_pre, d_f_pre, d_o_pre, d_g_pre], axis=1)

            d_W_x += step["x"].T @ d_pre
            d_W_h += step["h_prev"].T @ d_pre
            d_b += d_pre.sum(axis=0)

            d_h = d_pre @ self.W_h_.T
            d_c = d_c_prev

        return {
            "W_x": d_W_x,
            "W_h": d_W_h,
            "b": d_b,
            "W_hy": d_W_hy,
            "b_y": d_b_y,
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMClassifier":
        X = _check_sequences(X)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        K = len(self.classes_)
        if K < 2:
            raise ValueError("LSTMClassifier requires at least 2 classes.")

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
                probs, cache = self._forward(Xb)
                grads = self._backward(probs, Yb, cache)
                _clip_grads(grads)
                lr = self.learning_rate
                self.W_x_ -= lr * grads["W_x"]
                self.W_h_ -= lr * grads["W_h"]
                self.b_ -= lr * grads["b"]
                self.W_hy_ -= lr * grads["W_hy"]
                self.b_y_ -= lr * grads["b_y"]

            probs_all, _ = self._forward(X)
            eps = 1e-12
            loss = float(-np.mean(np.sum(Y_full * np.log(probs_all + eps), axis=1)))
            self.loss_curve_.append(loss)
            self.n_iter_ = epoch + 1

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["W_x_"])
        X = _check_sequences(X)
        probs, _ = self._forward(X)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class GRUClassifier(BaseEstimator, ClassifierMixin):
    """Many-to-one GRU for sequence classification.

    Update / reset / candidate gates; classify from final ``h_T`` via linear + softmax.

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
    W_xz_, W_hz_, b_z_, W_xr_, W_hr_, b_r_, W_xn_, W_hn_, b_n_
    W_hy_, b_y_
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
        sx = np.sqrt(1.0 / n_features)
        sh = np.sqrt(1.0 / h)
        self.W_xz_ = rng.standard_normal((n_features, h)) * sx
        self.W_hz_ = rng.standard_normal((h, h)) * sh
        self.b_z_ = np.zeros(h)
        self.W_xr_ = rng.standard_normal((n_features, h)) * sx
        self.W_hr_ = rng.standard_normal((h, h)) * sh
        self.b_r_ = np.zeros(h)
        self.W_xn_ = rng.standard_normal((n_features, h)) * sx
        self.W_hn_ = rng.standard_normal((h, h)) * sh
        self.b_n_ = np.zeros(h)
        self.W_hy_ = rng.standard_normal((h, n_classes)) * sh
        self.b_y_ = np.zeros(n_classes)

    def _forward(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
        N, T, _ = X.shape
        h = np.zeros((N, self.hidden_size))
        cache: list[dict[str, np.ndarray]] = []

        for t in range(T):
            x_t = X[:, t, :]
            h_prev = h
            z = activations.forward(
                "sigmoid", x_t @ self.W_xz_ + h_prev @ self.W_hz_ + self.b_z_
            )
            r = activations.forward(
                "sigmoid", x_t @ self.W_xr_ + h_prev @ self.W_hr_ + self.b_r_
            )
            rh = r * h_prev
            n = activations.forward(
                "tanh", x_t @ self.W_xn_ + rh @ self.W_hn_ + self.b_n_
            )
            h = (1.0 - z) * n + z * h_prev
            cache.append(
                {
                    "x": x_t,
                    "h_prev": h_prev,
                    "z": z,
                    "r": r,
                    "rh": rh,
                    "n": n,
                    "h": h,
                }
            )

        logits = h @ self.W_hy_ + self.b_y_
        probs = activations.softmax(logits)
        return probs, cache

    def _backward(
        self,
        probs: np.ndarray,
        Y: np.ndarray,
        cache: list[dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        n_batch = probs.shape[0]
        d_probs = (probs - Y) / n_batch
        d_W_hy = cache[-1]["h"].T @ d_probs
        d_b_y = d_probs.sum(axis=0)
        d_h = d_probs @ self.W_hy_.T

        zeros = {
            "W_xz": np.zeros_like(self.W_xz_),
            "W_hz": np.zeros_like(self.W_hz_),
            "b_z": np.zeros_like(self.b_z_),
            "W_xr": np.zeros_like(self.W_xr_),
            "W_hr": np.zeros_like(self.W_hr_),
            "b_r": np.zeros_like(self.b_r_),
            "W_xn": np.zeros_like(self.W_xn_),
            "W_hn": np.zeros_like(self.W_hn_),
            "b_n": np.zeros_like(self.b_n_),
            "W_hy": d_W_hy,
            "b_y": d_b_y,
        }

        for t in reversed(range(len(cache))):
            step = cache[t]
            z, r, n = step["z"], step["r"], step["n"]
            h_prev = step["h_prev"]

            d_z = d_h * (h_prev - n)
            d_n = d_h * (1.0 - z)
            d_h_prev = d_h * z

            d_n_pre = d_n * activations.grad("tanh", n)
            zeros["W_xn"] += step["x"].T @ d_n_pre
            zeros["W_hn"] += step["rh"].T @ d_n_pre
            zeros["b_n"] += d_n_pre.sum(axis=0)
            d_rh = d_n_pre @ self.W_hn_.T
            d_h_prev = d_h_prev + d_rh * r
            d_r = d_rh * h_prev

            d_r_pre = d_r * activations.grad("sigmoid", r)
            zeros["W_xr"] += step["x"].T @ d_r_pre
            zeros["W_hr"] += h_prev.T @ d_r_pre
            zeros["b_r"] += d_r_pre.sum(axis=0)
            d_h_prev = d_h_prev + d_r_pre @ self.W_hr_.T

            d_z_pre = d_z * activations.grad("sigmoid", z)
            zeros["W_xz"] += step["x"].T @ d_z_pre
            zeros["W_hz"] += h_prev.T @ d_z_pre
            zeros["b_z"] += d_z_pre.sum(axis=0)
            d_h = d_h_prev + d_z_pre @ self.W_hz_.T

        return zeros

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GRUClassifier":
        X = _check_sequences(X)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        K = len(self.classes_)
        if K < 2:
            raise ValueError("GRUClassifier requires at least 2 classes.")

        rng = check_random_state(self.random_state)
        self._init_params(X.shape[2], K, rng)
        Y_full = np.eye(K)[y_idx]
        n_samples = X.shape[0]
        self.loss_curve_ = []
        param_names = [
            "W_xz",
            "W_hz",
            "b_z",
            "W_xr",
            "W_hr",
            "b_r",
            "W_xn",
            "W_hn",
            "b_n",
            "W_hy",
            "b_y",
        ]

        for epoch in range(self.n_epochs):
            indices = np.arange(n_samples)
            rng.shuffle(indices)
            for start in range(0, n_samples, self.batch_size):
                batch = indices[start : start + self.batch_size]
                Xb, Yb = X[batch], Y_full[batch]
                probs, cache = self._forward(Xb)
                grads = self._backward(probs, Yb, cache)
                _clip_grads(grads)
                lr = self.learning_rate
                for name in param_names:
                    getattr(self, f"{name}_")[:] -= lr * grads[name]

            probs_all, _ = self._forward(X)
            eps = 1e-12
            loss = float(-np.mean(np.sum(Y_full * np.log(probs_all + eps), axis=1)))
            self.loss_curve_.append(loss)
            self.n_iter_ = epoch + 1

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["W_xz_"])
        X = _check_sequences(X)
        probs, _ = self._forward(X)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
