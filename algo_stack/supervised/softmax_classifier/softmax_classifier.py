"""Linear Softmax classifier — the canonical deep-learning classification head."""

from __future__ import annotations

import numpy as np

from algo_stack._base import BaseEstimator, ClassifierMixin
from algo_stack.utils import activations, optim
from algo_stack.utils.validation import check_array, check_random_state, check_X_y


class SoftmaxClassifier(BaseEstimator, ClassifierMixin):
    """Linear layer + softmax + cross-entropy, trained with mini-batch GD.

    This is the same model class taught in CS231n-style courses as the
    "Softmax classifier" — distinct from ``logistic_regression`` which uses
    full-batch GD and shares the GLM framing.

    Parameters
    ----------
    learning_rate : float, default 0.1
    optimizer : {"sgd", "momentum", "adam"}, default "momentum"
    n_epochs : int, default 200
    batch_size : int, default 32
    l2 : float, default 0.0
        L2 penalty on weights (bias not penalised).
    tol : float, default 1e-5
    shuffle : bool, default True
    random_state : int | None, default None
    fit_intercept : bool, default True

    Attributes
    ----------
    classes_ : ndarray
    coef_ : ndarray of shape (n_classes, n_features)
    intercept_ : ndarray of shape (n_classes,)
    loss_curve_ : list[float]
    n_iter_ : int
    """

    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        optimizer: str = "momentum",
        n_epochs: int = 200,
        batch_size: int = 32,
        l2: float = 0.0,
        tol: float = 1e-5,
        shuffle: bool = True,
        random_state: int | None = None,
        fit_intercept: bool = True,
    ) -> None:
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.tol = tol
        self.shuffle = shuffle
        self.random_state = random_state
        self.fit_intercept = fit_intercept

    def _params(self) -> list[np.ndarray]:
        return [self.coef_, self.intercept_]

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted(["coef_"])
        X = check_array(X)
        return X @ self.coef_.T + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return activations.softmax(self.decision_function(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftmaxClassifier":
        X, y = check_X_y(X, y, y_numeric=False)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        K = len(self.classes_)
        if K < 2:
            raise ValueError("SoftmaxClassifier requires at least 2 classes.")

        n_samples, n_features = X.shape
        rng = check_random_state(self.random_state)
        scale = np.sqrt(1.0 / n_features)
        self.coef_ = rng.standard_normal((K, n_features)) * scale * 0.01
        self.intercept_ = np.zeros(K)

        opt = optim.make_optimizer(self.optimizer, learning_rate=self.learning_rate)
        opt.reset(self._params())

        Y_full = np.eye(K)[y_idx]
        self.loss_curve_: list[float] = []
        prev_loss: float | None = None

        for epoch in range(self.n_epochs):
            indices = np.arange(n_samples)
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, n_samples, self.batch_size):
                batch = indices[start : start + self.batch_size]
                Xb, Yb = X[batch], Y_full[batch]
                n_b = Xb.shape[0]
                logits = Xb @ self.coef_.T + self.intercept_
                probs = activations.softmax(logits)
                delta = (probs - Yb) / n_b
                grad_W = delta.T @ Xb
                grad_b = delta.sum(axis=0)
                if self.l2:
                    grad_W += (self.l2 / n_b) * self.coef_
                opt.step(self._params(), [grad_W, grad_b])

            probs_all = activations.softmax(X @ self.coef_.T + self.intercept_)
            eps = 1e-12
            loss = float(-np.mean(np.sum(Y_full * np.log(probs_all + eps), axis=1)))
            if self.l2:
                loss += 0.5 * self.l2 * float(np.sum(self.coef_ * self.coef_)) / n_samples
            self.loss_curve_.append(loss)
            self.n_iter_ = epoch + 1
            if prev_loss is not None and abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

        return self
