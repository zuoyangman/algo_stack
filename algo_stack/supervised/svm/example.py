"""Runnable demo: ``python -m algo_stack.supervised.svm.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.supervised.svm import SVC, SVR
from algo_stack.utils.preprocessing import train_test_split


def main() -> None:
    rng = np.random.default_rng(0)

    print("--- Binary SVC (linear / RBF) on two blobs ---")
    X0 = rng.normal(loc=[-2.0, 0.0], scale=0.6, size=(80, 2))
    X1 = rng.normal(loc=[2.0, 0.0], scale=0.6, size=(80, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 80 + [1] * 80)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    for kernel in ("linear", "rbf"):
        clf = SVC(C=1.0, kernel=kernel, max_iter=200, random_state=0).fit(Xtr, ytr)
        print(
            f"kernel={kernel:6s}  n_sv={len(clf.support_vectors_):3d}  "
            f"train={clf.score(Xtr, ytr):.3f}  test={clf.score(Xte, yte):.3f}"
        )

    print("\n--- Multiclass SVC (OvR, RBF) ---")
    centres = np.array([[-3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
    Xm = np.vstack([c + rng.normal(scale=0.5, size=(50, 2)) for c in centres])
    ym = np.repeat(np.arange(3), 50)
    Xtr, Xte, ytr, yte = train_test_split(Xm, ym, test_size=0.25, random_state=1)
    clf = SVC(C=1.0, kernel="rbf", max_iter=300, random_state=0).fit(Xtr, ytr)
    print(f"test_acc={clf.score(Xte, yte):.3f}  n_sv={len(clf.support_vectors_)}")

    print("\n--- SVR (Kernel Ridge dual) on noisy sine ---")
    Xs = rng.uniform(-3, 3, size=(120, 1))
    ys = np.sin(Xs.ravel()) + 0.15 * rng.normal(size=Xs.shape[0])
    Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=0.25, random_state=0)
    for C in (0.5, 5.0, 50.0):
        reg = SVR(C=C, kernel="rbf").fit(Xtr, ytr)
        print(f"C={C:5.1f}  train_R²={reg.score(Xtr, ytr):.3f}  test_R²={reg.score(Xte, yte):.3f}")


if __name__ == "__main__":
    main()
