"""Runnable demo: ``python -m algo_stack.supervised.logistic_regression.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.supervised.logistic_regression import LogisticRegression
from algo_stack.utils.preprocessing import StandardScaler, train_test_split


def make_blobs(rng: np.random.Generator, n_per_class: int, centres: np.ndarray):
    K, d = centres.shape
    X = np.vstack(
        [centres[k] + rng.normal(scale=0.7, size=(n_per_class, d)) for k in range(K)]
    )
    y = np.repeat(np.arange(K), n_per_class)
    return X, y


def main() -> None:
    rng = np.random.default_rng(0)

    print("--- Binary classification ---")
    centres2 = np.array([[-2.0, 0.0], [2.0, 0.0]])
    X, y = make_blobs(rng, 200, centres2)
    X = StandardScaler().fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)

    clf = LogisticRegression(learning_rate=0.5, n_iter=500).fit(X_tr, y_tr)
    print(f"coef_={np.round(clf.coef_, 3)}, intercept_={clf.intercept_:.3f}")
    print(f"Train accuracy = {clf.score(X_tr, y_tr):.3f}")
    print(f"Test  accuracy = {clf.score(X_te, y_te):.3f}")
    print(f"Final train loss = {clf.loss_history_[-1]:.4f}, n_iter_ = {clf.n_iter_}")

    print("\n--- Multinomial (3-class) classification ---")
    centres3 = np.array([[-3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
    X, y = make_blobs(rng, 150, centres3)
    X = StandardScaler().fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)

    clf = LogisticRegression(learning_rate=0.5, n_iter=800).fit(X_tr, y_tr)
    print(f"classes_ = {clf.classes_}")
    print(f"coef_.shape = {clf.coef_.shape}, intercept_.shape = {clf.intercept_.shape}")
    print(f"Train accuracy = {clf.score(X_tr, y_tr):.3f}")
    print(f"Test  accuracy = {clf.score(X_te, y_te):.3f}")


if __name__ == "__main__":
    main()
