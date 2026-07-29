"""Runnable demo: ``python -m algo_stack.optimization.lbfgs.example``."""

from __future__ import annotations

import numpy as np

from algo_stack.optimization.lbfgs import minimize_lbfgs


def main() -> None:
    # Classic Rosenbrock banana function.
    def fun(z: np.ndarray) -> float:
        x, y = z
        return float((1 - x) ** 2 + 100 * (y - x**2) ** 2)

    def jac(z: np.ndarray) -> np.ndarray:
        x, y = z
        return np.array(
            [
                -2 * (1 - x) - 400 * x * (y - x**2),
                200 * (y - x**2),
            ],
            dtype=np.float64,
        )

    result = minimize_lbfgs(fun, np.array([-1.2, 1.0]), jac=jac, max_iter=200, tol=1e-8)
    print(f"success={result.success}  n_iter={result.n_iter}  f={result.fun:.2e}")
    print(f"x* ≈ {np.round(result.x, 6)}  (true optimum is [1, 1])")
    print(f"||grad||_∞ = {np.max(np.abs(result.grad)):.2e}")
    print(f"message: {result.message}")


if __name__ == "__main__":
    main()
