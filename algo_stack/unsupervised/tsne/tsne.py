"""t-distributed Stochastic Neighbor Embedding (exact / dense variant)."""

from __future__ import annotations

import numpy as np

from algo_stack._base import BaseEstimator, TransformerMixin
from algo_stack.utils.validation import check_array, check_random_state


def _pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    xx = np.sum(X * X, axis=1)[:, None]
    d2 = xx + xx.T - 2.0 * (X @ X.T)
    np.maximum(d2, 0.0, out=d2)
    np.fill_diagonal(d2, 0.0)
    return d2


def _binary_search_beta(
    d2_row: np.ndarray, target_entropy: float, max_iter: int = 50, tol: float = 1e-5
) -> np.ndarray:
    """Find precision β such that H(P_i·) ≈ log(perplexity). Return P_i·."""

    beta_min, beta_max = -np.inf, np.inf
    beta = 1.0
    n = d2_row.shape[0]
    for _ in range(max_iter):
        # Softmax over other points (self already zeroed).
        log_p = -beta * d2_row
        log_p -= np.max(log_p)
        p = np.exp(log_p)
        sum_p = p.sum()
        if sum_p == 0:
            p = np.full(n, 1.0 / n)
            break
        p /= sum_p
        # Shannon entropy in nats.
        entropy = -np.sum(p * np.log(np.maximum(p, 1e-12)))
        diff = entropy - target_entropy
        if abs(diff) < tol:
            break
        if diff > 0:
            # Entropy too high → sharpen (increase β).
            beta_min = beta
            beta = beta * 2.0 if np.isinf(beta_max) else 0.5 * (beta + beta_max)
        else:
            beta_max = beta
            beta = beta / 2.0 if np.isinf(beta_min) else 0.5 * (beta + beta_min)
    return p


def _joint_probabilities(X: np.ndarray, perplexity: float) -> np.ndarray:
    n = X.shape[0]
    d2 = _pairwise_sq_dists(X)
    target = np.log(perplexity)
    P = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        # Mask self by setting distance to +inf so softmax ignores it.
        row = d2[i].copy()
        row[i] = np.inf
        P[i] = _binary_search_beta(row, target)
        P[i, i] = 0.0
    # Symmetrise.
    P = (P + P.T) / (2.0 * n)
    np.maximum(P, 1e-12, out=P)
    return P


class TSNE(BaseEstimator, TransformerMixin):
    """Exact t-SNE for small datasets (n ≲ 200).

    Parameters
    ----------
    n_components : int, default 2
    perplexity : float, default 30.0
    learning_rate : float, default 200.0
    n_iter : int, default 1000
    early_exaggeration : float, default 12.0
    random_state : int | Generator | None, default None

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
    kl_divergence_ : float
    n_iter_ : int
    """

    def __init__(
        self,
        *,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        early_exaggeration: float = 12.0,
        random_state: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.early_exaggeration = early_exaggeration
        self.random_state = random_state

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        X = check_array(X)
        n = X.shape[0]
        if self.n_components < 1:
            raise ValueError("n_components must be >= 1.")
        if self.perplexity >= n:
            raise ValueError(
                f"perplexity ({self.perplexity}) must be < n_samples ({n})."
            )

        rng = check_random_state(self.random_state)
        P = _joint_probabilities(X, self.perplexity)
        P_exag = P * self.early_exaggeration

        Y = rng.normal(scale=1e-4, size=(n, self.n_components))
        dY = np.zeros_like(Y)
        iY = np.zeros_like(Y)
        gains = np.ones_like(Y)

        min_gain = 0.01
        momentum = 0.5
        final_momentum = 0.8
        mom_switch_iter = 250
        exaggeration_iter = 250

        for it in range(self.n_iter):
            # Student-t affinities in low-d.
            d2 = _pairwise_sq_dists(Y)
            num = 1.0 / (1.0 + d2)
            np.fill_diagonal(num, 0.0)
            Q = num / np.maximum(num.sum(), 1e-12)
            np.maximum(Q, 1e-12, out=Q)

            PQ = (P_exag if it < exaggeration_iter else P) - Q
            for i in range(n):
                dY[i] = np.sum((PQ[:, i] * num[:, i])[:, None] * (Y[i] - Y), axis=0)

            # Adaptive gains + momentum (as in van der Maaten's reference).
            gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * (
                (dY > 0) == (iY > 0)
            )
            gains = np.maximum(gains, min_gain)
            mom = final_momentum if it >= mom_switch_iter else momentum
            iY = mom * iY - self.learning_rate * gains * dY
            Y = Y + iY
            Y -= Y.mean(axis=0)

        # Final KL.
        d2 = _pairwise_sq_dists(Y)
        num = 1.0 / (1.0 + d2)
        np.fill_diagonal(num, 0.0)
        Q = num / np.maximum(num.sum(), 1e-12)
        np.maximum(Q, 1e-12, out=Q)
        kl = float(np.sum(P * np.log(P / Q)))

        self.embedding_ = Y
        self.kl_divergence_ = kl
        self.n_iter_ = self.n_iter
        return Y

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "TSNE":
        self.fit_transform(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """t-SNE does not support out-of-sample transform; return training embedding."""

        self._check_is_fitted(["embedding_"])
        raise RuntimeError(
            "TSNE does not support transforming new data; use fit_transform on the "
            "full dataset, or access `.embedding_` after fit."
        )
