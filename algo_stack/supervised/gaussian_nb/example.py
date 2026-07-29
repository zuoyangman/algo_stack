"""Runnable demo: ``python -m algo_stack.supervised.gaussian_nb.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.supervised.gaussian_nb import GaussianNB
from algo_stack.utils.preprocessing import train_test_split


def main() -> None:
    rng = np.random.default_rng(0)
    centres = np.array([[-2.0, -1.0], [2.0, 1.0], [0.0, 3.0]])
    n = 100
    X = np.vstack([c + rng.normal(scale=0.7, size=(n, 2)) for c in centres])
    y = np.repeat(np.arange(3), n)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
    clf = GaussianNB().fit(X_tr, y_tr)
    print("class_prior_:", np.round(clf.class_prior_, 3))
    print("theta_:\n", np.round(clf.theta_, 3))
    print(f"Train accuracy = {clf.score(X_tr, y_tr):.3f}")
    print(f"Test  accuracy = {clf.score(X_te, y_te):.3f}")


if __name__ == "__main__":
    main()
