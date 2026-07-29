"""Runnable demo: ``python -m algo_stack.metrics.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    mean_squared_error,
    precision_recall_f1,
    r2_score,
    silhouette_score,
)


def main() -> None:
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    print("accuracy =", accuracy_score(y_true, y_pred))
    print("precision, recall, f1 =", precision_recall_f1(y_true, y_pred))
    print("confusion_matrix =\n", confusion_matrix(y_true, y_pred))

    y_prob = np.array([0.1, 0.8, 0.4, 0.3, 0.9, 0.6])
    print("log_loss =", round(log_loss(y_true, y_prob), 4))

    y_reg = np.array([1.0, 2.0, 3.0])
    y_hat = np.array([1.1, 1.9, 3.2])
    print("mse =", round(mean_squared_error(y_reg, y_hat), 4))
    print("r2  =", round(r2_score(y_reg, y_hat), 4))

    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(size=(30, 2)), rng.normal(size=(30, 2)) + 4.0])
    labels = np.array([0] * 30 + [1] * 30)
    print("silhouette =", round(silhouette_score(X, labels), 4))


if __name__ == "__main__":
    main()
