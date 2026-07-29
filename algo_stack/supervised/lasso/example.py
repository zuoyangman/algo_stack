"""Runnable demo: ``python -m algo_stack.supervised.lasso.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.supervised.lasso import Lasso
from algo_stack.utils.preprocessing import train_test_split


def main() -> None:
    rng = np.random.default_rng(0)
    n_samples, n_features = 250, 20
    true_w = np.zeros(n_features)
    true_w[:4] = [3.0, -2.0, 1.5, -1.0]

    X = rng.normal(size=(n_samples, n_features))
    y = X @ true_w + 0.2 * rng.normal(size=n_samples)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    model = Lasso(alpha=0.05, max_iter=2000).fit(X_train, y_train)
    print("True non-zeros:     ", np.flatnonzero(true_w))
    print("Estimated non-zeros:", np.flatnonzero(np.abs(model.coef_) > 1e-3))
    print("Estimated coef_[:8]:", np.round(model.coef_[:8], 3))
    print(f"n_iter_ = {model.n_iter_}")
    print(f"Train R² = {model.score(X_train, y_train):.4f}")
    print(f"Test  R² = {model.score(X_test, y_test):.4f}")


if __name__ == "__main__":
    main()
