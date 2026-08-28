"""Runnable demo: ``python -m algo_stack.supervised.mlp.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.supervised.mlp import MLPClassifier, MLPRegressor
from algo_stack.utils.preprocessing import StandardScaler, train_test_split


def main() -> None:
    rng = np.random.default_rng(0)

    print("--- MLP regression: noisy sine ---")
    X = rng.uniform(-3, 3, size=(400, 1))
    y = np.sin(X.ravel() * 2) + 0.1 * rng.normal(size=X.shape[0])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    reg = MLPRegressor(
        hidden_layer_sizes=(32, 32),
        activation="tanh",
        learning_rate=0.01,
        momentum=0.9,
        batch_size=32,
        n_epochs=300,
        tol=1e-7,
        random_state=0,
    ).fit(Xtr_s, ytr)
    print(f"epochs={reg.n_iter_}  final_loss={reg.loss_curve_[-1]:.4f}")
    print(f"train R² = {reg.score(Xtr_s, ytr):.3f}")
    print(f"test  R² = {reg.score(Xte_s, yte):.3f}")

    print("\n--- MLP classification: 2-D XOR-like 4-blob problem ---")
    centres = np.array([[-2.0, -2.0], [2.0, 2.0], [-2.0, 2.0], [2.0, -2.0]])
    labels  = np.array([0, 0, 1, 1])
    X = np.vstack([c + rng.normal(scale=0.4, size=(100, 2)) for c in centres])
    y = np.repeat(labels, 100)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

    clf = MLPClassifier(
        hidden_layer_sizes=(16, 16),
        activation="relu",
        learning_rate=0.05,
        momentum=0.9,
        batch_size=32,
        n_epochs=200,
        random_state=0,
    ).fit(Xtr, ytr)
    print(f"epochs={clf.n_iter_}  final_loss={clf.loss_curve_[-1]:.4f}")
    print(f"train acc = {clf.score(Xtr, ytr):.3f}")
    print(f"test  acc = {clf.score(Xte, yte):.3f}")


if __name__ == "__main__":
    main()
