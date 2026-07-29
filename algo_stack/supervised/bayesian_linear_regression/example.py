"""Runnable demo: ``python -m algo_stack.supervised.bayesian_linear_regression.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.supervised.bayesian_linear_regression import BayesianLinearRegression
from algo_stack.utils.preprocessing import train_test_split


def main() -> None:
    rng = np.random.default_rng(0)
    n_samples, n_features = 150, 3
    true_w = np.array([1.5, -2.0, 0.5])
    true_b = 1.0
    noise_std = 0.3

    X = rng.normal(size=(n_samples, n_features))
    y = X @ true_w + true_b + noise_std * rng.normal(size=n_samples)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    # β ≈ 1 / σ²
    model = BayesianLinearRegression(alpha=1.0, beta=1.0 / noise_std**2).fit(
        X_train, y_train
    )
    print("Estimated coef_:    ", np.round(model.coef_, 3))
    print("True coef:          ", true_w)
    print("Estimated intercept:", round(model.intercept_, 3))
    print(f"alpha_={model.alpha_}, beta_={model.beta_}")
    print(f"Train R² = {model.score(X_train, y_train):.4f}")
    print(f"Test  R² = {model.score(X_test, y_test):.4f}")
    std = model.predict_std(X_test)
    print(f"Mean predictive std on test: {std.mean():.3f}")


if __name__ == "__main__":
    main()
