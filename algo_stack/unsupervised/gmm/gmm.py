"""Gaussian Mixture Model fitted by Expectation-Maximisation."""

from __future__ import annotations

import numpy as np

from algo_stack._base import BaseEstimator, ClusterMixin
from algo_stack.unsupervised.kmeans.kmeans import _kmeans_pp_init
from algo_stack.utils.validation import check_array, check_random_state


def _logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    if axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


class GaussianMixture(BaseEstimator, ClusterMixin):
    """Finite Gaussian mixture estimated with EM.

    Parameters
    ----------
    n_components : int, default 1
    max_iter : int, default 100
    tol : float, default 1e-3
        Absolute change in average log-likelihood for convergence.
    covariance_type : {"diag", "full"}, default "diag"
    reg_covar : float, default 1e-6
        Diagonal jitter added to every covariance for numerical stability.
    random_state : int | Generator | None, default None

    Attributes
    ----------
    weights_ : ndarray of shape (n_components,)
    means_ : ndarray of shape (n_components, n_features)
    covariances_ : ndarray
        Shape ``(n_components, n_features)`` for ``diag``, or
        ``(n_components, n_features, n_features)`` for ``full``.
    responsibilities_ : ndarray of shape (n_samples, n_components)
    n_iter_ : int
    lower_bound_ : float
        Average log-likelihood of the training data under the fitted model.
    """

    def __init__(
        self,
        *,
        n_components: int = 1,
        max_iter: int = 100,
        tol: float = 1e-3,
        covariance_type: str = "diag",
        reg_covar: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.random_state = random_state

    def _init_params(self, X: np.ndarray, rng: np.random.Generator) -> None:
        n_samples, n_features = X.shape
        k = self.n_components
        # k-means++ seeding spreads means and avoids collapsing both into one blob.
        self.means_ = _kmeans_pp_init(X, k, rng)
        self.weights_ = np.full(k, 1.0 / k)
        if self.covariance_type == "diag":
            var = np.var(X, axis=0) + self.reg_covar
            self.covariances_ = np.tile(var, (k, 1))
        elif self.covariance_type == "full":
            cov = np.cov(X, rowvar=False) + self.reg_covar * np.eye(n_features)
            if cov.ndim == 0:
                cov = np.array([[float(cov)]])
            self.covariances_ = np.tile(cov, (k, 1, 1))
        else:
            raise ValueError(
                f"covariance_type must be 'diag' or 'full', got {self.covariance_type!r}."
            )

    def _log_gauss(self, X: np.ndarray) -> np.ndarray:
        """Log density of each component for every sample. Shape (n, k)."""

        n_samples, n_features = X.shape
        k = self.n_components
        log_prob = np.empty((n_samples, k))
        if self.covariance_type == "diag":
            for j in range(k):
                var = self.covariances_[j]
                diff = X - self.means_[j]
                log_det = np.sum(np.log(var))
                quad = np.sum(diff * diff / var, axis=1)
                log_prob[:, j] = -0.5 * (
                    n_features * np.log(2.0 * np.pi) + log_det + quad
                )
        else:
            for j in range(k):
                cov = self.covariances_[j]
                # Cholesky for log-det and quadratic form.
                L = np.linalg.cholesky(cov)
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
                diff = X - self.means_[j]
                sol = np.linalg.solve(L, diff.T).T
                quad = np.sum(sol * sol, axis=1)
                log_prob[:, j] = -0.5 * (
                    n_features * np.log(2.0 * np.pi) + log_det + quad
                )
        return log_prob

    def _e_step(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        log_resp = self._log_gauss(X) + np.log(self.weights_)
        log_norm = _logsumexp(log_resp, axis=1)
        resp = np.exp(log_resp - log_norm[:, None])
        lower_bound = float(np.mean(log_norm))
        return resp, lower_bound

    def _m_step(self, X: np.ndarray, resp: np.ndarray) -> None:
        n_samples, n_features = X.shape
        nk = resp.sum(axis=0) + 10.0 * np.finfo(float).eps
        self.weights_ = nk / n_samples
        self.means_ = (resp.T @ X) / nk[:, None]

        if self.covariance_type == "diag":
            cov = np.empty((self.n_components, n_features))
            for j in range(self.n_components):
                diff = X - self.means_[j]
                cov[j] = (resp[:, j] @ (diff * diff)) / nk[j] + self.reg_covar
            self.covariances_ = cov
        else:
            cov = np.empty((self.n_components, n_features, n_features))
            for j in range(self.n_components):
                diff = X - self.means_[j]
                weighted = diff * np.sqrt(resp[:, j])[:, None]
                cov[j] = (weighted.T @ weighted) / nk[j]
                cov[j].flat[:: n_features + 1] += self.reg_covar
            self.covariances_ = cov

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "GaussianMixture":
        if self.n_components < 1:
            raise ValueError("n_components must be >= 1.")
        X = check_array(X)
        if X.shape[0] < self.n_components:
            raise ValueError(
                f"n_samples={X.shape[0]} < n_components={self.n_components}."
            )
        rng = check_random_state(self.random_state)
        self._init_params(X, rng)

        lower_bound = -np.inf
        for it in range(self.max_iter):
            resp, new_lb = self._e_step(X)
            self._m_step(X, resp)
            if abs(new_lb - lower_bound) < self.tol:
                lower_bound = new_lb
                break
            lower_bound = new_lb

        self.responsibilities_ = resp
        self.n_iter_ = it + 1
        self.lower_bound_ = lower_bound
        self.labels_ = np.argmax(resp, axis=1)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Per-sample log-likelihood under the mixture."""

        self._check_is_fitted(["means_", "weights_", "covariances_"])
        X = check_array(X)
        log_prob = self._log_gauss(X) + np.log(self.weights_)
        return _logsumexp(log_prob, axis=1)

    def score(self, X: np.ndarray, y: np.ndarray | None = None) -> float:
        return float(np.mean(self.score_samples(X)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["means_", "weights_", "covariances_"])
        X = check_array(X)
        resp, _ = self._e_step(X)
        return resp

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
