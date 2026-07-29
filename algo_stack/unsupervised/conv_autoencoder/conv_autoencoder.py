"""Convolutional Autoencoder for small single-channel images.

Architecture (default 8×8 grayscale)::

    Conv → ReLU → MaxPool → Flatten → Dense (bottleneck)
    Dense → reshape → nearest upsample → Conv → ReLU → Conv (1 ch)
"""

from __future__ import annotations

import numpy as np

from algo_stack._base import BaseEstimator, TransformerMixin
from algo_stack.supervised.cnn.cnn import (
    _check_images,
    _conv2d_backward,
    _conv2d_forward,
    _maxpool2d_backward,
    _maxpool2d_forward,
)
from algo_stack.utils import activations
from algo_stack.utils.validation import check_random_state


def _pad2d(X: np.ndarray, pad: int) -> np.ndarray:
    if pad == 0:
        return X
    return np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")


def _crop2d(X: np.ndarray, pad: int) -> np.ndarray:
    if pad == 0:
        return X
    return X[:, :, pad:-pad, pad:-pad]


def _conv_same_forward(
    X: np.ndarray, W: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, dict]:
    """3×3 (or k×k) convolution with zero-padding to preserve H×W."""

    kH, kW = W.shape[2], W.shape[3]
    pad_h, pad_w = kH // 2, kW // 2
    # Assume square odd kernels for same-pad.
    pad = pad_h
    X_pad = _pad2d(X, pad)
    out, cache = _conv2d_forward(X_pad, W, b, stride=1)
    cache["pad"] = pad
    cache["X_unpadded"] = X
    return out, cache


def _conv_same_backward(
    d_out: np.ndarray, cache: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dX_pad, dW, db = _conv2d_backward(d_out, cache)
    pad = cache["pad"]
    dX = _crop2d(dX_pad, pad)
    return dX, dW, db


def _upsample_nearest(X: np.ndarray, scale: int = 2) -> tuple[np.ndarray, dict]:
    out = X.repeat(scale, axis=2).repeat(scale, axis=3)
    return out, {"scale": scale, "in_shape": X.shape}


def _upsample_nearest_backward(d_out: np.ndarray, cache: dict) -> np.ndarray:
    scale = cache["scale"]
    N, C, H, W = cache["in_shape"]
    # Sum gradients over each scale×scale block.
    d = d_out.reshape(N, C, H, scale, W, scale).sum(axis=(3, 5))
    return d


class ConvAutoencoder(BaseEstimator, TransformerMixin):
    """Toy convolutional autoencoder for ``(N, 1, H, W)`` grayscale images.

    Encoder: padded Conv → ReLU → MaxPool → Flatten → Dense bottleneck.
    Decoder: Dense → reshape → nearest upsample → Conv → ReLU → Conv (1 ch).

    Parameters
    ----------
    encoding_dim : int, default 8
    num_filters : int, default 4
    filter_size : int, default 3
        Odd kernel size (same-padding).
    pool_size : int, default 2
    learning_rate : float, default 0.05
    n_epochs : int, default 80
    batch_size : int, default 16
    random_state : int | None, default None

    Attributes
    ----------
    enc_conv_W_, enc_conv_b_
    enc_fc_W_, enc_fc_b_
    dec_fc_W_, dec_fc_b_
    dec_conv1_W_, dec_conv1_b_
    dec_conv2_W_, dec_conv2_b_
    loss_curve_, n_iter_
    image_shape_ : tuple
        ``(C, H, W)`` of training inputs.
    """

    def __init__(
        self,
        *,
        encoding_dim: int = 8,
        num_filters: int = 4,
        filter_size: int = 3,
        pool_size: int = 2,
        learning_rate: float = 0.05,
        n_epochs: int = 80,
        batch_size: int = 16,
        random_state: int | None = None,
    ) -> None:
        self.encoding_dim = encoding_dim
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def _init_params(
        self, C: int, H: int, W: int, rng: np.random.Generator
    ) -> None:
        if self.filter_size % 2 == 0:
            raise ValueError("filter_size must be odd for same-padding.")
        if H % self.pool_size != 0 or W % self.pool_size != 0:
            raise ValueError(
                f"Image size {(H, W)} must be divisible by pool_size={self.pool_size}."
            )

        k = self.filter_size
        F = self.num_filters
        e = self.encoding_dim
        pool_h, pool_w = H // self.pool_size, W // self.pool_size
        flat_dim = F * pool_h * pool_w

        self.image_shape_ = (C, H, W)
        self._pool_h_ = pool_h
        self._pool_w_ = pool_w
        self._flat_dim_ = flat_dim

        scale_c = np.sqrt(2.0 / (C * k * k))
        self.enc_conv_W_ = rng.standard_normal((F, C, k, k)) * scale_c
        self.enc_conv_b_ = np.zeros(F)
        self.enc_fc_W_ = rng.standard_normal((flat_dim, e)) * np.sqrt(2.0 / flat_dim)
        self.enc_fc_b_ = np.zeros(e)

        self.dec_fc_W_ = rng.standard_normal((e, flat_dim)) * np.sqrt(2.0 / e)
        self.dec_fc_b_ = np.zeros(flat_dim)
        self.dec_conv1_W_ = rng.standard_normal((F, F, k, k)) * np.sqrt(2.0 / (F * k * k))
        self.dec_conv1_b_ = np.zeros(F)
        self.dec_conv2_W_ = rng.standard_normal((C, F, k, k)) * np.sqrt(2.0 / (F * k * k))
        self.dec_conv2_b_ = np.zeros(C)

    def _params(self) -> list[np.ndarray]:
        return [
            self.enc_conv_W_,
            self.enc_conv_b_,
            self.enc_fc_W_,
            self.enc_fc_b_,
            self.dec_fc_W_,
            self.dec_fc_b_,
            self.dec_conv1_W_,
            self.dec_conv1_b_,
            self.dec_conv2_W_,
            self.dec_conv2_b_,
        ]

    def _encode_forward(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        conv, c_conv = _conv_same_forward(X, self.enc_conv_W_, self.enc_conv_b_)
        relu = activations.forward("relu", conv)
        pooled, c_pool = _maxpool2d_forward(
            relu, self.pool_size, self.pool_size
        )
        flat = pooled.reshape(pooled.shape[0], -1)
        code = flat @ self.enc_fc_W_ + self.enc_fc_b_
        cache = {
            "c_conv": c_conv,
            "relu": relu,
            "c_pool": c_pool,
            "pool_shape": pooled.shape,
            "flat": flat,
        }
        return code, cache

    def _decode_forward(self, code: np.ndarray) -> tuple[np.ndarray, dict]:
        N = code.shape[0]
        F = self.num_filters
        pre = code @ self.dec_fc_W_ + self.dec_fc_b_
        spatial = pre.reshape(N, F, self._pool_h_, self._pool_w_)
        up, c_up = _upsample_nearest(spatial, scale=self.pool_size)
        conv1, c1 = _conv_same_forward(up, self.dec_conv1_W_, self.dec_conv1_b_)
        relu1 = activations.forward("relu", conv1)
        recon, c2 = _conv_same_forward(relu1, self.dec_conv2_W_, self.dec_conv2_b_)
        cache = {
            "code": code,
            "pre": pre,
            "spatial": spatial,
            "c_up": c_up,
            "up": up,
            "c1": c1,
            "relu1": relu1,
            "c2": c2,
        }
        return recon, cache

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict, dict]:
        code, enc_cache = self._encode_forward(X)
        recon, dec_cache = self._decode_forward(code)
        return recon, code, enc_cache, dec_cache

    def _backward(
        self,
        X: np.ndarray,
        recon: np.ndarray,
        enc_cache: dict,
        dec_cache: dict,
    ) -> list[np.ndarray]:
        n = X.shape[0]
        d_recon = 2.0 * (recon - X) / n

        # Decoder conv2
        d_relu1, d_dec_conv2_W, d_dec_conv2_b = _conv_same_backward(
            d_recon, dec_cache["c2"]
        )
        d_conv1 = d_relu1 * activations.grad("relu", dec_cache["relu1"])
        d_up, d_dec_conv1_W, d_dec_conv1_b = _conv_same_backward(
            d_conv1, dec_cache["c1"]
        )
        d_spatial = _upsample_nearest_backward(d_up, dec_cache["c_up"])
        d_pre = d_spatial.reshape(n, -1)
        d_dec_fc_W = dec_cache["code"].T @ d_pre
        d_dec_fc_b = d_pre.sum(axis=0)
        d_code = d_pre @ self.dec_fc_W_.T

        # Encoder FC
        d_enc_fc_W = enc_cache["flat"].T @ d_code
        d_enc_fc_b = d_code.sum(axis=0)
        d_flat = d_code @ self.enc_fc_W_.T
        d_pool = d_flat.reshape(enc_cache["pool_shape"])
        d_relu = _maxpool2d_backward(d_pool, enc_cache["c_pool"])
        d_conv = d_relu * activations.grad("relu", enc_cache["relu"])
        _, d_enc_conv_W, d_enc_conv_b = _conv_same_backward(d_conv, enc_cache["c_conv"])

        return [
            d_enc_conv_W,
            d_enc_conv_b,
            d_enc_fc_W,
            d_enc_fc_b,
            d_dec_fc_W,
            d_dec_fc_b,
            d_dec_conv1_W,
            d_dec_conv1_b,
            d_dec_conv2_W,
            d_dec_conv2_b,
        ]

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "ConvAutoencoder":
        X = _check_images(X)
        if X.shape[1] != 1:
            raise ValueError("ConvAutoencoder expects single-channel images (C=1).")
        rng = check_random_state(self.random_state)
        _, C, H, W = X.shape
        self._init_params(C, H, W, rng)

        n_samples = X.shape[0]
        self.loss_curve_ = []
        params = self._params()

        for epoch in range(self.n_epochs):
            indices = np.arange(n_samples)
            rng.shuffle(indices)
            for start in range(0, n_samples, self.batch_size):
                batch = indices[start : start + self.batch_size]
                Xb = X[batch]
                recon, _, enc_c, dec_c = self._forward(Xb)
                grads = self._backward(Xb, recon, enc_c, dec_c)
                total_norm = float(np.sqrt(sum(np.sum(g * g) for g in grads)))
                if total_norm > 5.0:
                    scale = 5.0 / total_norm
                    grads = [g * scale for g in grads]
                lr = self.learning_rate
                for p, g in zip(params, grads):
                    p -= lr * g

            recon_all, _, _, _ = self._forward(X)
            loss = float(np.mean((recon_all - X) ** 2))
            self.loss_curve_.append(loss)
            self.n_iter_ = epoch + 1

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return bottleneck codes of shape ``(n_samples, encoding_dim)``."""

        self._check_is_fitted(["enc_conv_W_"])
        X = _check_images(X)
        code, _ = self._encode_forward(X)
        return code

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Encode then decode; return reconstructions with shape of ``X``."""

        self._check_is_fitted(["dec_conv2_W_"])
        X = _check_images(X)
        recon, _, _, _ = self._forward(X)
        return recon
