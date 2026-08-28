"""Logistic regression with L2 penalty (binary + multinomial), batch GD.

This is a didactic, NumPy-only reference implementation. For an industrial
solver you would use L-BFGS / IRLS; we use full-batch gradient descent so the
training loop stays one line of math.
"""

from __future__ import annotations

import numpy as np

from algo_stack._base import BaseEstimator, ClassifierMixin
from algo_stack.utils.validation import check_array, check_X_y


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # numerically stable sigmoid
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _softmax(z: np.ndarray) -> np.ndarray:
    # subtract max for numerical stability
    z_shift = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z_shift)
    return ez / ez.sum(axis=1, keepdims=True)


class LogisticRegression(BaseEstimator, ClassifierMixin):
    """Logistic regression with optional L2 penalty.

    Automatically switches between binary (sigmoid + BCE) and multinomial
    (softmax + cross-entropy) depending on the number of classes in ``y``.

    Parameters
    ----------
    learning_rate : float, default 0.1
        Step size for full-batch gradient descent.
    n_iter : int, default 1000
        Maximum number of gradient steps.
    l2 : float, default 0.0
        L2 (ridge) regularisation strength applied to the weights only (the
        bias is not penalised).
    tol : float, default 1e-6
        Stop early if `||grad||_∞ < tol`.
    fit_intercept : bool, default True
    random_state : int | Generator | None, default None
        Only used to break ties when initialising weights to a tiny random
        perturbation (currently we start from zeros, so this is reserved).

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
    coef_ : ndarray of shape (n_classes, n_features) for multinomial, or
            (n_features,) for binary.
    intercept_ : ndarray of shape (n_classes,) or float for binary.
    n_iter_ : int
        Number of iterations actually run.
    loss_history_ : list[float]
        Mean log-loss after every iteration (useful for sanity checks).
    """

    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        n_iter: int = 1000,
        l2: float = 0.0,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.l2 = l2
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        X, y = check_X_y(X, y, y_numeric=False)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        if n_classes < 2:
            raise ValueError("LogisticRegression needs at least 2 classes in y.")

        self._multinomial_ = n_classes > 2
        if self._multinomial_:
            W = np.zeros((n_classes, n_features))
            b = np.zeros(n_classes)
            Y_oh = np.eye(n_classes)[y_idx]  # one-hot, shape (n, K)
        else:
            W = np.zeros(n_features)
            b = 0.0
            y_bin = y_idx.astype(np.float64)  # 0 / 1

        loss_history: list[float] = []
        n_iter_run = 0

        for it in range(self.n_iter):
            if self._multinomial_:
                logits = X @ W.T + b  # shape (n, K)
                P = _softmax(logits)
                # cross-entropy loss (mean over samples)
                eps = 1e-12
                ll = -np.mean(np.sum(Y_oh * np.log(P + eps), axis=1))
                if self.l2:
                    ll += 0.5 * self.l2 * float(np.sum(W * W)) / n_samples
                loss_history.append(ll)

                grad_logits = (P - Y_oh) / n_samples  # (n, K)
                grad_W = grad_logits.T @ X            # (K, d)
                if self.l2:
                    grad_W += (self.l2 / n_samples) * W
                grad_b = grad_logits.sum(axis=0)      # (K,)

                inf_norm = max(np.abs(grad_W).max(), np.abs(grad_b).max())
                W -= self.learning_rate * grad_W
                if self.fit_intercept:
                    b -= self.learning_rate * grad_b
                n_iter_run = it + 1
                if inf_norm < self.tol:
                    break
            else:
                logits = X @ W + b
                p = _sigmoid(logits)
                eps = 1e-12
                ll = -np.mean(
                    y_bin * np.log(p + eps) + (1.0 - y_bin) * np.log(1.0 - p + eps)
                )
                if self.l2:
                    ll += 0.5 * self.l2 * float(W @ W) / n_samples
                loss_history.append(ll)

                err = (p - y_bin) / n_samples
                grad_W = X.T @ err
                if self.l2:
                    grad_W += (self.l2 / n_samples) * W
                grad_b = float(err.sum())

                inf_norm = max(np.abs(grad_W).max(), abs(grad_b))
                W -= self.learning_rate * grad_W
                if self.fit_intercept:
                    b -= self.learning_rate * grad_b
                n_iter_run = it + 1
                if inf_norm < self.tol:
                    break

        self.coef_ = W
        self.intercept_ = b if self._multinomial_ else float(b)
        self.n_iter_ = n_iter_run
        self.loss_history_ = loss_history
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["coef_"])
        X = check_array(X)
        if self._multinomial_:
            return X @ self.coef_.T + self.intercept_
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        if self._multinomial_:
            return _softmax(scores)
        p1 = _sigmoid(scores)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
