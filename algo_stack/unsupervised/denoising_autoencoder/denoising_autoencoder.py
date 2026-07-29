"""Denoising Autoencoder: reconstruct clean inputs from corrupted observations."""

from __future__ import annotations

import numpy as np

from algo_stack._base import BaseEstimator, TransformerMixin
from algo_stack.utils import activations, optim
from algo_stack.utils.validation import check_array, check_random_state


class DenoisingAutoencoder(BaseEstimator, TransformerMixin):
    """Autoencoder trained to reconstruct clean ``X`` from Gaussian-corrupted inputs.

    During training, inputs are corrupted as ``X̃ = X + ε``, ``ε ~ N(0, noise_std²)``.
    The reconstruction target remains the clean ``X``.

    Parameters
    ----------
    encoding_dim : int, default 4
        Bottleneck size.
    hidden_dim : int, default 16
        Width of encoder/decoder hidden layers.
    noise_std : float, default 0.1
        Std of isotropic Gaussian corruption applied during training.
    learning_rate : float, default 0.001
    n_epochs : int, default 200
    batch_size : int, default 32
    random_state : int | None, default None

    Attributes
    ----------
    encoder_W1_, encoder_b1_, encoder_W2_, encoder_b2_
    decoder_W1_, decoder_b1_, decoder_W2_, decoder_b2_
    loss_curve_, n_iter_
    """

    def __init__(
        self,
        *,
        encoding_dim: int = 4,
        hidden_dim: int = 16,
        noise_std: float = 0.1,
        learning_rate: float = 0.001,
        n_epochs: int = 200,
        batch_size: int = 32,
        random_state: int | None = None,
    ) -> None:
        self.encoding_dim = encoding_dim
        self.hidden_dim = hidden_dim
        self.noise_std = noise_std
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def _params(self) -> list[np.ndarray]:
        return [
            self.encoder_W1_,
            self.encoder_b1_,
            self.encoder_W2_,
            self.encoder_b2_,
            self.decoder_W1_,
            self.decoder_b1_,
            self.decoder_W2_,
            self.decoder_b2_,
        ]

    def _init_params(self, n_features: int, rng: np.random.Generator) -> None:
        h, e = self.hidden_dim, self.encoding_dim
        s1 = np.sqrt(1.0 / n_features)
        s2 = np.sqrt(1.0 / h)
        s3 = np.sqrt(1.0 / e)
        self.encoder_W1_ = rng.standard_normal((n_features, h)) * s1
        self.encoder_b1_ = np.zeros(h)
        self.encoder_W2_ = rng.standard_normal((h, e)) * s2
        self.encoder_b2_ = np.zeros(e)
        self.decoder_W1_ = rng.standard_normal((e, h)) * s3
        self.decoder_b1_ = np.zeros(h)
        self.decoder_W2_ = rng.standard_normal((h, n_features)) * s2
        self.decoder_b2_ = np.zeros(n_features)

    def _encode(self, X: np.ndarray) -> np.ndarray:
        a1 = activations.forward("relu", X @ self.encoder_W1_ + self.encoder_b1_)
        return a1 @ self.encoder_W2_ + self.encoder_b2_

    def _forward(self, X_in: np.ndarray, X_target: np.ndarray) -> tuple[np.ndarray, dict]:
        enc_pre = X_in @ self.encoder_W1_ + self.encoder_b1_
        enc_h = activations.forward("relu", enc_pre)
        code = enc_h @ self.encoder_W2_ + self.encoder_b2_
        dec_pre = code @ self.decoder_W1_ + self.decoder_b1_
        dec_h = activations.forward("relu", dec_pre)
        recon = dec_h @ self.decoder_W2_ + self.decoder_b2_
        cache = {
            "X_in": X_in,
            "X_target": X_target,
            "enc_pre": enc_pre,
            "enc_h": enc_h,
            "code": code,
            "dec_pre": dec_pre,
            "dec_h": dec_h,
        }
        return recon, cache

    def _backward(self, recon: np.ndarray, cache: dict) -> list[np.ndarray]:
        X_in = cache["X_in"]
        X_target = cache["X_target"]
        n = X_in.shape[0]
        d_recon = 2.0 * (recon - X_target) / n

        d_dec_h = d_recon @ self.decoder_W2_.T
        d_dec_W2 = cache["dec_h"].T @ d_recon
        d_dec_b2 = d_recon.sum(axis=0)

        d_dec_pre = d_dec_h * activations.grad("relu", cache["dec_h"])
        d_dec_W1 = cache["code"].T @ d_dec_pre
        d_dec_b1 = d_dec_pre.sum(axis=0)

        d_code = d_dec_pre @ self.decoder_W1_.T
        d_enc_h = d_code @ self.encoder_W2_.T
        d_enc_W2 = cache["enc_h"].T @ d_code
        d_enc_b2 = d_code.sum(axis=0)

        d_enc_pre = d_enc_h * activations.grad("relu", cache["enc_h"])
        d_enc_W1 = X_in.T @ d_enc_pre
        d_enc_b1 = d_enc_pre.sum(axis=0)

        return [
            d_enc_W1,
            d_enc_b1,
            d_enc_W2,
            d_enc_b2,
            d_dec_W1,
            d_dec_b1,
            d_dec_W2,
            d_dec_b2,
        ]

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "DenoisingAutoencoder":
        X = check_array(X)
        rng = check_random_state(self.random_state)
        self._init_params(X.shape[1], rng)
        opt = optim.make_optimizer("momentum", learning_rate=self.learning_rate)
        opt.reset(self._params())
        n_samples = X.shape[0]
        self.loss_curve_ = []

        for epoch in range(self.n_epochs):
            indices = np.arange(n_samples)
            rng.shuffle(indices)
            for start in range(0, n_samples, self.batch_size):
                batch = indices[start : start + self.batch_size]
                Xb = X[batch]
                noise = rng.normal(scale=self.noise_std, size=Xb.shape)
                X_noisy = Xb + noise
                recon, cache = self._forward(X_noisy, Xb)
                grads = self._backward(recon, cache)
                total_norm = float(np.sqrt(sum(np.sum(g * g) for g in grads)))
                if total_norm > 5.0:
                    scale = 5.0 / total_norm
                    grads = [g * scale for g in grads]
                opt.step(self._params(), grads)

            # Evaluate reconstruction of clean inputs (no noise).
            recon_all, _ = self._forward(X, X)
            loss = float(np.mean((recon_all - X) ** 2))
            self.loss_curve_.append(loss)
            self.n_iter_ = epoch + 1

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return the bottleneck code (encoded representation)."""

        self._check_is_fitted(["encoder_W1_"])
        X = check_array(X)
        return self._encode(X)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Decode the bottleneck and return reconstructed inputs (no noise)."""

        self._check_is_fitted(["decoder_W1_"])
        X = check_array(X)
        recon, _ = self._forward(X, X)
        return recon
