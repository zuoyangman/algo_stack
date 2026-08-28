"""Toy 2-D Convolutional Neural Network (im2col conv + max-pool + linear head).

Designed for small grayscale images ``(n_samples, 1, height, width)`` — the
classic teaching stack from CS231n.
"""

from __future__ import annotations

import numpy as np

from algo_stack._base import BaseEstimator, ClassifierMixin
from algo_stack.utils import activations
from algo_stack.utils.validation import check_random_state


def _check_images(X: np.ndarray, name: str = "X") -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 4:
        raise ValueError(
            f"Expected {name} with shape (n_samples, channels, height, width), "
            f"got ndim={arr.ndim}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf.")
    return arr


def _im2col(
    X: np.ndarray, kernel_h: int, kernel_w: int, stride: int = 1
) -> tuple[np.ndarray, int, int]:
    """Unfold image patches into columns (im2col).

    Returns ``(cols, out_h, out_w)`` where ``cols`` has shape
    ``(C·kH·kW, N·out_h·out_w)``.
    """

    N, C, H, W = X.shape
    out_h = (H - kernel_h) // stride + 1
    out_w = (W - kernel_w) // stride + 1
    cols = np.zeros((C * kernel_h * kernel_w, N * out_h * out_w), dtype=X.dtype)
    col = 0
    for oy in range(out_h):
        for ox in range(out_w):
            iy, ix = oy * stride, ox * stride
            patch = X[:, :, iy : iy + kernel_h, ix : ix + kernel_w]
            # Each column is one sample's flattened patch: (C·kH·kW, N).
            cols[:, col : col + N] = patch.reshape(N, -1).T
            col += N
    return cols, out_h, out_w


def _col2im(
    cols: np.ndarray,
    X_shape: tuple[int, int, int, int],
    kernel_h: int,
    kernel_w: int,
    stride: int,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Scatter-add columns back into an image (conv backward)."""

    N, C, H, W = X_shape
    X_grad = np.zeros(X_shape, dtype=cols.dtype)
    col = 0
    for oy in range(out_h):
        for ox in range(out_w):
            iy, ix = oy * stride, ox * stride
            patch = cols[:, col : col + N].T.reshape(N, C, kernel_h, kernel_w)
            X_grad[:, :, iy : iy + kernel_h, ix : ix + kernel_w] += patch
            col += N
    return X_grad


def _conv2d_forward(
    X: np.ndarray, W: np.ndarray, b: np.ndarray, stride: int = 1
) -> tuple[np.ndarray, dict]:
    """2-D convolution via im2col."""

    N, C_in, _, _ = X.shape
    C_out, _, kH, kW = W.shape
    cols, out_h, out_w = _im2col(X, kH, kW, stride)
    W_col = W.reshape(C_out, -1)
    raw = W_col @ cols + b[:, None]  # (C_out, N·out_h·out_w)
    # Columns are laid out as (oy, ox, n): position-major, sample-minor.
    out = raw.reshape(C_out, out_h, out_w, N).transpose(3, 0, 1, 2)
    cache = {
        "X": X,
        "W": W,
        "cols": cols,
        "out_h": out_h,
        "out_w": out_w,
        "stride": stride,
        "kH": kH,
        "kW": kW,
    }
    return out, cache


def _conv2d_backward(
    d_out: np.ndarray, cache: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, W, cols = cache["X"], cache["W"], cache["cols"]
    C_out, _, kH, kW = W.shape
    N = X.shape[0]
    out_h, out_w = cache["out_h"], cache["out_w"]
    stride = cache["stride"]

    d_out_flat = d_out.transpose(1, 2, 3, 0).reshape(C_out, -1)
    dW = (d_out_flat @ cols.T).reshape(W.shape)
    db = d_out_flat.sum(axis=1)
    W_col = W.reshape(C_out, -1)
    d_cols = W_col.T @ d_out_flat
    dX = _col2im(d_cols, X.shape, kH, kW, stride, out_h, out_w)
    return dX, dW, db


def _maxpool2d_forward(
    X: np.ndarray, pool_h: int, pool_w: int, stride: int | None = None
) -> tuple[np.ndarray, dict]:
    stride = pool_h if stride is None else stride
    N, C, H, W = X.shape
    out_h = (H - pool_h) // stride + 1
    out_w = (W - pool_w) // stride + 1
    out = np.zeros((N, C, out_h, out_w), dtype=X.dtype)
    argmax = np.zeros((N, C, out_h, out_w, 2), dtype=np.int64)
    for oy in range(out_h):
        for ox in range(out_w):
            iy, ix = oy * stride, ox * stride
            window = X[:, :, iy : iy + pool_h, ix : ix + pool_w]
            flat = window.reshape(N, C, -1)
            idx = np.argmax(flat, axis=2)
            out[:, :, oy, ox] = np.take_along_axis(
                flat, idx[:, :, None], axis=2
            ).squeeze(2)
            iy_off = idx // pool_w
            ix_off = idx % pool_w
            argmax[:, :, oy, ox, 0] = iy + iy_off
            argmax[:, :, oy, ox, 1] = ix + ix_off
    cache = {
        "X_shape": X.shape,
        "argmax": argmax,
        "pool_h": pool_h,
        "pool_w": pool_w,
        "stride": stride,
        "out_h": out_h,
        "out_w": out_w,
    }
    return out, cache


def _maxpool2d_backward(d_out: np.ndarray, cache: dict) -> np.ndarray:
    X_shape = cache["X_shape"]
    argmax = cache["argmax"]
    pool_h, pool_w = cache["pool_h"], cache["pool_w"]
    stride = cache["stride"]
    out_h, out_w = cache["out_h"], cache["out_w"]
    dX = np.zeros(X_shape, dtype=d_out.dtype)
    N, C = X_shape[0], X_shape[1]
    for oy in range(out_h):
        for ox in range(out_w):
            for n in range(N):
                for c in range(C):
                    iy = int(argmax[n, c, oy, ox, 0])
                    ix = int(argmax[n, c, oy, ox, 1])
                    dX[n, c, iy, ix] += d_out[n, c, oy, ox]
    return dX


class CNNClassifier(BaseEstimator, ClassifierMixin):
    """Toy CNN for small single-channel images.

    Architecture: Conv → ReLU → MaxPool → Flatten → Linear → Softmax.

    Parameters
    ----------
    num_filters : int, default 4
    filter_size : int, default 3
    pool_size : int, default 2
    learning_rate : float, default 0.05
    n_epochs : int, default 80
    batch_size : int, default 16
    random_state : int | None, default None

    Attributes
    ----------
    classes_ : ndarray
    conv_W_, conv_b_ : convolution weights / bias
    fc_W_, fc_b_ : fully-connected head
    loss_curve_ : list[float]
    n_iter_ : int
    """

    def __init__(
        self,
        *,
        num_filters: int = 4,
        filter_size: int = 3,
        pool_size: int = 2,
        learning_rate: float = 0.05,
        n_epochs: int = 80,
        batch_size: int = 16,
        random_state: int | None = None,
    ) -> None:
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def _init_params(self, n_classes: int, rng: np.random.Generator) -> None:
        k = self.filter_size
        scale = np.sqrt(2.0 / (1 * k * k))
        self.conv_W_ = rng.standard_normal((self.num_filters, 1, k, k)) * scale
        self.conv_b_ = np.zeros(self.num_filters)
        self._n_classes_ = n_classes
        self._fc_in_dim_: int | None = None
        self.fc_W_: np.ndarray | None = None
        self.fc_b_: np.ndarray | None = None

    def _ensure_fc(self, flat_dim: int, rng: np.random.Generator) -> None:
        if self.fc_W_ is None:
            self._fc_in_dim_ = flat_dim
            self.fc_W_ = rng.standard_normal((flat_dim, self._n_classes_)) * np.sqrt(
                2.0 / flat_dim
            )
            self.fc_b_ = np.zeros(self._n_classes_)

    def _forward(
        self, X: np.ndarray, rng: np.random.Generator | None = None
    ) -> tuple[np.ndarray, list]:
        caches: list[dict] = []
        conv_out, c1 = _conv2d_forward(X, self.conv_W_, self.conv_b_)
        caches.append({"type": "conv", "cache": c1})
        relu_out = activations.forward("relu", conv_out)
        caches.append({"type": "relu", "a": relu_out})
        pool_out, c2 = _maxpool2d_forward(relu_out, self.pool_size, self.pool_size)
        caches.append({"type": "pool", "cache": c2})
        flat = pool_out.reshape(pool_out.shape[0], -1)
        caches.append({"type": "flat", "shape": pool_out.shape})
        if self.fc_W_ is None:
            fc_rng = np.random.default_rng(
                int(rng.integers(0, 2**31 - 1)) if rng is not None else 0
            )
            self._ensure_fc(flat.shape[1], fc_rng)
        logits = flat @ self.fc_W_ + self.fc_b_
        caches.append({"type": "fc", "flat": flat})
        probs = activations.softmax(logits)
        return probs, caches

    def _backward(
        self, caches: list, probs: np.ndarray, Y: np.ndarray
    ) -> dict[str, np.ndarray]:
        n = probs.shape[0]
        delta = (probs - Y) / n
        grads: dict[str, np.ndarray] = {}

        fc_cache = caches[-1]
        flat = fc_cache["flat"]
        grads["fc_W"] = flat.T @ delta
        grads["fc_b"] = delta.sum(axis=0)
        d_flat = delta @ self.fc_W_.T

        pool_shape = caches[3]["shape"]
        d_pool = d_flat.reshape(pool_shape)

        pool_cache = caches[2]["cache"]
        d_relu = _maxpool2d_backward(d_pool, pool_cache)

        relu_a = caches[1]["a"]
        d_conv = d_relu * activations.grad("relu", relu_a)

        conv_cache = caches[0]["cache"]
        _, d_conv_W, d_conv_b = _conv2d_backward(d_conv, conv_cache)
        grads["conv_W"] = d_conv_W
        grads["conv_b"] = d_conv_b
        return grads

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CNNClassifier":
        X = _check_images(X)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        K = len(self.classes_)
        if K < 2:
            raise ValueError("CNNClassifier requires at least 2 classes.")

        rng = check_random_state(self.random_state)
        self._init_params(K, rng)
        Y_full = np.eye(K)[y_idx]
        n_samples = X.shape[0]
        self.loss_curve_ = []

        # Initialise FC layer with a dry-run forward pass.
        self._forward(X[:1], rng=rng)

        for epoch in range(self.n_epochs):
            indices = np.arange(n_samples)
            rng.shuffle(indices)
            for start in range(0, n_samples, self.batch_size):
                batch = indices[start : start + self.batch_size]
                Xb, Yb = X[batch], Y_full[batch]
                probs, caches = self._forward(Xb)
                grads = self._backward(caches, probs, Yb)
                lr = self.learning_rate
                self.conv_W_ -= lr * grads["conv_W"]
                self.conv_b_ -= lr * grads["conv_b"]
                self.fc_W_ -= lr * grads["fc_W"]
                self.fc_b_ -= lr * grads["fc_b"]

            probs_all, _ = self._forward(X)
            eps = 1e-12
            loss = float(-np.mean(np.sum(Y_full * np.log(probs_all + eps), axis=1)))
            self.loss_curve_.append(loss)
            self.n_iter_ = epoch + 1

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["conv_W_"])
        X = _check_images(X)
        probs, _ = self._forward(X)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
