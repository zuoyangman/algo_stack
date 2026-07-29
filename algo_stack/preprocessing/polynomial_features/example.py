"""Runnable demo: ``python -m algo_stack.preprocessing.polynomial_features.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.preprocessing.polynomial_features import PolynomialFeatures


def main() -> None:
    X = np.array([[2.0, 3.0]])
    poly = PolynomialFeatures(degree=2, include_bias=True).fit(X)
    print("powers_ =\n", poly.powers_)
    print("expanded =", poly.transform(X))
    # Expected: [1, 2, 3, 4, 6, 9]

    poly_ix = PolynomialFeatures(degree=2, interaction_only=True).fit(X)
    print("interaction_only =", poly_ix.transform(X))
    # Expected: [1, 2, 3, 6]


if __name__ == "__main__":
    main()
