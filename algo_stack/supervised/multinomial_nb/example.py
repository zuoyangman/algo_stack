"""Runnable demo: ``python -m algo_stack.supervised.multinomial_nb.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.supervised.multinomial_nb import MultinomialNB
from algo_stack.utils.preprocessing import train_test_split


def main() -> None:
    rng = np.random.default_rng(0)
    # Two topics with different preferred tokens.
    n, d = 200, 30
    X0 = rng.poisson(lam=np.linspace(3, 0.2, d), size=(n, d)).astype(float)
    X1 = rng.poisson(lam=np.linspace(0.2, 3, d), size=(n, d)).astype(float)
    X = np.vstack([X0, X1])
    y = np.array([0] * n + [1] * n)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
    clf = MultinomialNB(alpha=1.0).fit(X_tr, y_tr)
    print("classes_:", clf.classes_)
    print("class_log_prior_:", np.round(clf.class_log_prior_, 3))
    print(f"Train accuracy = {clf.score(X_tr, y_tr):.3f}")
    print(f"Test  accuracy = {clf.score(X_te, y_te):.3f}")


if __name__ == "__main__":
    main()
