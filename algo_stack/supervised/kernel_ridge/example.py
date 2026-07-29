"""Runnable demo: ``python -m algo_stack.supervised.kernel_ridge.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.supervised.kernel_ridge import KernelRidge
from algo_stack.utils.preprocessing import train_test_split


def main() -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, size=(150, 1))
    y = np.sin(X.ravel()) + 0.1 * rng.normal(size=X.shape[0])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

    print("--- KernelRidge on noisy sine ---")
    for kernel, kwargs in (
        ("linear", {}),
        ("rbf", {"alpha": 0.1}),
        ("polynomial", {"alpha": 0.1, "degree": 3}),
    ):
        reg = KernelRidge(kernel=kernel, **kwargs).fit(Xtr, ytr)
        print(
            f"kernel={kernel:10s}  "
            f"train_R²={reg.score(Xtr, ytr):.3f}  test_R²={reg.score(Xte, yte):.3f}"
        )


if __name__ == "__main__":
    main()
