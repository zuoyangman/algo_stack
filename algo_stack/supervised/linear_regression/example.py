"""Runnable demo: ``python -m algo_stack.supervised.linear_regression.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.supervised.linear_regression import LinearRegression
from algo_stack.utils.preprocessing import train_test_split


def main() -> None:
    rng = np.random.default_rng(0)
    n_samples, n_features = 200, 3
    true_w = np.array([1.5, -2.0, 0.5])
    true_b = 4.0

    X = rng.normal(size=(n_samples, n_features))
    noise = 0.1 * rng.normal(size=n_samples)
    y = X @ true_w + true_b + noise

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    model = LinearRegression().fit(X_train, y_train)
    print("Estimated coef_:    ", np.round(model.coef_, 3))
    print("True coef:          ", true_w)
    print("Estimated intercept:", round(model.intercept_, 3))
    print("True intercept:     ", true_b)
    print(f"Train R² = {model.score(X_train, y_train):.4f}")
    print(f"Test  R² = {model.score(X_test, y_test):.4f}")


if __name__ == "__main__":
    main()
