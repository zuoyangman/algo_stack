"""Runnable demo: ``python -m algo_stack.preprocessing.minmax_scaler.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.preprocessing.minmax_scaler import MinMaxScaler


def main() -> None:
    X = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
    sc = MinMaxScaler(feature_range=(0, 1)).fit(X)
    Z = sc.transform(X)
    print("data_min_ =", sc.data_min_)
    print("data_max_ =", sc.data_max_)
    print("scaled =\n", Z)
    print("round-trip =\n", sc.inverse_transform(Z))


if __name__ == "__main__":
    main()
