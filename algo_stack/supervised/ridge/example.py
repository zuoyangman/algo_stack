"""Runnable demo: ``python -m algo_stack.supervised.ridge.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.supervised.ridge import Ridge
from algo_stack.utils.preprocessing import train_test_split


def main() -> None:
    rng = np.random.default_rng(0)
    n_samples, n_features = 200, 5
    true_w = np.array([1.5, -2.0, 0.5, 0.0, 0.0])
    true_b = 3.0

    X = rng.normal(size=(n_samples, n_features))
    y = X @ true_w + true_b + 0.1 * rng.normal(size=n_samples)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    model = Ridge(alpha=1.0).fit(X_train, y_train)
    print("Estimated coef_:    ", np.round(model.coef_, 3))
    print("True coef:          ", true_w)
    print("Estimated intercept:", round(model.intercept_, 3))
    print(f"Train R² = {model.score(X_train, y_train):.4f}")
    print(f"Test  R² = {model.score(X_test, y_test):.4f}")

    # Larger alpha → smaller coefficient norm
    big = Ridge(alpha=100.0).fit(X_train, y_train)
    print(f"||w|| @ α=1:   {np.linalg.norm(model.coef_):.3f}")
    print(f"||w|| @ α=100: {np.linalg.norm(big.coef_):.3f}")


if __name__ == "__main__":
    main()
