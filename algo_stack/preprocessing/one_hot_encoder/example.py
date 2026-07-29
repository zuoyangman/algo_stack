"""Runnable demo: ``python -m algo_stack.preprocessing.one_hot_encoder.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.preprocessing.one_hot_encoder import OneHotEncoder


def main() -> None:
    X = np.array([["red"], ["blue"], ["red"], ["green"]])
    enc = OneHotEncoder().fit(X)
    print("categories_ =", [c.tolist() for c in enc.categories_])
    print("encoded =\n", enc.transform(X))

    enc_drop = OneHotEncoder(drop="first").fit(X)
    print("drop='first' encoded =\n", enc_drop.transform(X))


if __name__ == "__main__":
    main()
