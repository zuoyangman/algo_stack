"""Variational Autoencoder with reparameterisation and β-weighted ELBO."""

from __future__ import annotations

import numpy as np

from algo_stack._base import BaseEstimator, TransformerMixin
from algo_stack.utils import activations, optim
from algo_stack.utils.validation import check_array, check_random_state


class VariationalAutoencoder(BaseEstimator, TransformerMixin):
    """β-VAE: encoder → (μ, log σ²), reparameterised latent, decoder.

    Minimises reconstruction MSE plus ``β · KL(N(μ, σ²) || N(0, I))``.

    Parameters
    ----------
    encoding_dim : int, default 4
        Latent dimensionality.
    hidden_dim : int, default 16
        Width of encoder / decoder hidden layers.
    learning_rate : float, default 0.001
    n_epochs : int, default 200
    batch_size : int, default 32
    beta : float, default 1.0
        Weight on the KL term (β-VAE).
    random_state : int | None, default None

    Attributes
    ----------
    encoder_W1_, encoder_b1_, W_mu_, b_mu_, W_logvar_, b_logvar_
    decoder_W1_, decoder_b1_, decoder_W2_, decoder_b2_
    loss_curve_, n_iter_, n_features_in_
    """

    def __init__(
        self,
        *,
        encoding_dim: int = 4,
        hidden_dim: int = 16,
        learning_rate: float = 0.001,
        n_epochs: int = 200,
        batch_size: int = 32,
        beta: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        self.encoding_dim = encoding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.beta = beta
        self.random_state = random_state

    def _params(self) -> list[np.ndarray]:
        return [
            self.encoder_W1_,
            self.encoder_b1_,
            self.W_mu_,
            self.b_mu_,
            self.W_logvar_,
            self.b_logvar_,
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
        self.W_mu_ = rng.standard_normal((h, e)) * s2
        self.b_mu_ = np.zeros(e)
        self.W_logvar_ = rng.standard_normal((h, e)) * s2
        self.b_logvar_ = np.zeros(e)
        self.decoder_W1_ = rng.standard_normal((e, h)) * s3
        self.decoder_b1_ = np.zeros(h)
        self.decoder_W2_ = rng.standard_normal((h, n_features)) * s2
        self.decoder_b2_ = np.zeros(n_features)
        self.n_features_in_ = n_features

    def _encode_stats(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        enc_pre = X @ self.encoder_W1_ + self.encoder_b1_
        enc_h = activations.forward("relu", enc_pre)
        mu = enc_h @ self.W_mu_ + self.b_mu_
        log_var = enc_h @ self.W_logvar_ + self.b_logvar_
        return enc_h, mu, log_var

    def _decode(self, Z: np.ndarray) -> np.ndarray:
        dec_pre = Z @ self.decoder_W1_ + self.decoder_b1_
        dec_h = activations.forward("relu", dec_pre)
        return dec_h @ self.decoder_W2_ + self.decoder_b2_

    def _forward(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict]:
        enc_pre = X @ self.encoder_W1_ + self.encoder_b1_
        enc_h = activations.forward("relu", enc_pre)
        mu = enc_h @ self.W_mu_ + self.b_mu_
        log_var = enc_h @ self.W_logvar_ + self.b_logvar_
        # Clamp for numerical stability.
        log_var = np.clip(log_var, -10.0, 10.0)
        std = np.exp(0.5 * log_var)
        eps = rng.standard_normal(size=mu.shape)
        z = mu + std * eps
        dec_pre = z @ self.decoder_W1_ + self.decoder_b1_
        dec_h = activations.forward("relu", dec_pre)
        recon = dec_h @ self.decoder_W2_ + self.decoder_b2_
        cache = {
            "X": X,
            "enc_pre": enc_pre,
            "enc_h": enc_h,
            "mu": mu,
            "log_var": log_var,
            "std": std,
            "eps": eps,
            "z": z,
            "dec_pre": dec_pre,
            "dec_h": dec_h,
        }
        return recon, cache

    def _elbo_parts(
        self, recon: np.ndarray, X: np.ndarray, mu: np.ndarray, log_var: np.ndarray
    ) -> tuple[float, float]:
        recon_loss = float(np.mean((recon - X) ** 2))
        # KL per sample, then mean: 0.5 * Σ (μ² + σ² − 1 − log σ²)
        kl = 0.5 * np.mean(np.sum(mu**2 + np.exp(log_var) - 1.0 - log_var, axis=1))
        return recon_loss, float(kl)

    def _backward(self, recon: np.ndarray, cache: dict) -> list[np.ndarray]:
        X = cache["X"]
        n = X.shape[0]
        mu = cache["mu"]
        log_var = cache["log_var"]
        std = cache["std"]
        eps = cache["eps"]

        d_recon = 2.0 * (recon - X) / n

        d_dec_h = d_recon @ self.decoder_W2_.T
        d_dec_W2 = cache["dec_h"].T @ d_recon
        d_dec_b2 = d_recon.sum(axis=0)

        d_dec_pre = d_dec_h * activations.grad("relu", cache["dec_h"])
        d_dec_W1 = cache["z"].T @ d_dec_pre
        d_dec_b1 = d_dec_pre.sum(axis=0)
        d_z = d_dec_pre @ self.decoder_W1_.T

        # KL gradients (averaged like recon via /n on sums).
        # L_kl = (β / n) * 0.5 * Σ_i Σ_j (μ² + exp(log_var) − 1 − log_var)
        # ∂L_kl/∂μ = (β / n) * μ
        # ∂L_kl/∂log_var = (β / n) * 0.5 * (exp(log_var) − 1)
        d_mu = d_z + (self.beta / n) * mu
        d_log_var = d_z * (0.5 * std * eps) + (self.beta / n) * 0.5 * (
            np.exp(log_var) - 1.0
        )

        d_enc_h = d_mu @ self.W_mu_.T + d_log_var @ self.W_logvar_.T
        d_W_mu = cache["enc_h"].T @ d_mu
        d_b_mu = d_mu.sum(axis=0)
        d_W_logvar = cache["enc_h"].T @ d_log_var
        d_b_logvar = d_log_var.sum(axis=0)

        d_enc_pre = d_enc_h * activations.grad("relu", cache["enc_h"])
        d_enc_W1 = X.T @ d_enc_pre
        d_enc_b1 = d_enc_pre.sum(axis=0)

        return [
            d_enc_W1,
            d_enc_b1,
            d_W_mu,
            d_b_mu,
            d_W_logvar,
            d_b_logvar,
            d_dec_W1,
            d_dec_b1,
            d_dec_W2,
            d_dec_b2,
        ]

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "VariationalAutoencoder":
        X = check_array(X)
        rng = check_random_state(self.random_state)
        self._init_params(X.shape[1], rng)
        opt = optim.make_optimizer("adam", learning_rate=self.learning_rate)
        opt.reset(self._params())
        n_samples = X.shape[0]
        self.loss_curve_ = []

        for epoch in range(self.n_epochs):
            indices = np.arange(n_samples)
            rng.shuffle(indices)
            for start in range(0, n_samples, self.batch_size):
                batch = indices[start : start + self.batch_size]
                Xb = X[batch]
                recon, cache = self._forward(Xb, rng)
                grads = self._backward(recon, cache)
                total_norm = float(np.sqrt(sum(np.sum(g * g) for g in grads)))
                if total_norm > 5.0:
                    scale = 5.0 / total_norm
                    grads = [g * scale for g in grads]
                opt.step(self._params(), grads)

            recon_all, cache_all = self._forward(X, rng)
            recon_loss, kl = self._elbo_parts(
                recon_all, X, cache_all["mu"], cache_all["log_var"]
            )
            self.loss_curve_.append(recon_loss + self.beta * kl)
            self.n_iter_ = epoch + 1

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return latent means ``μ`` (deterministic encoding)."""

        self._check_is_fitted(["W_mu_"])
        X = check_array(X)
        _, mu, _ = self._encode_stats(X)
        return mu

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Encode to ``μ`` and decode (no sampling)."""

        self._check_is_fitted(["decoder_W1_"])
        X = check_array(X)
        _, mu, _ = self._encode_stats(X)
        return self._decode(mu)

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Draw ``z ~ N(0, I)`` and decode."""

        self._check_is_fitted(["decoder_W1_"])
        rng = check_random_state(self.random_state)
        z = rng.standard_normal((n_samples, self.encoding_dim))
        return self._decode(z)
