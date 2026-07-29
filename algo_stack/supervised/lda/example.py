"""Runnable demo: ``python -m algo_stack.supervised.lda.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.supervised.lda import LinearDiscriminantAnalysis
from algo_stack.utils.preprocessing import train_test_split


def make_blobs(rng, n_per_class, centres, scale=0.8):
    X = np.vstack([c + rng.normal(scale=scale, size=(n_per_class, 2)) for c in centres])
    y = np.repeat(np.arange(len(centres)), n_per_class)
    return X, y


def main() -> None:
    rng = np.random.default_rng(0)
    centres = np.array([[-2.5, 0.0], [0.0, 2.5], [2.5, 0.0]])
    X, y = make_blobs(rng, 120, centres)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)

    clf = LinearDiscriminantAnalysis().fit(X_tr, y_tr)
    print("classes_:", clf.classes_)
    print("priors_:", np.round(clf.priors_, 3))
    print("means_:\n", np.round(clf.means_, 3))
    print(f"Train accuracy = {clf.score(X_tr, y_tr):.3f}")
    print(f"Test  accuracy = {clf.score(X_te, y_te):.3f}")


if __name__ == "__main__":
    main()
