# Ridge Regression — Extension Guide

## 1. Extension surface

- **`alpha`**: replace the scalar with a vector (per-feature penalties) or a
  schedule.
- **Solver**: swap `np.linalg.solve` for Cholesky, conjugate gradient, or SVD
  (`w = V (s² / (s² + α)) Uᵀ y` via the SVD of `X`).
- **Kernel Ridge**: replace `X Xᵀ` with a Gram matrix `K` — see a sibling
  `KernelRidge` class rather than overloading this one.

## 2. Common variants

### 2.1 Weighted Ridge

Pass `sample_weight` into `fit`, multiply rows of `X` and `y` by
`√w_i` before forming the normal equations.

### 2.2 Multi-output Ridge

Solve against a matrix `Y ∈ ℝ^{n×k}`; `coef_` becomes `(n_features, k)`.

### 2.3 Cross-validated `alpha`

Wrap with a grid search over `alpha` using held-out R² / MSE.

## 3. Invariants

- `fit` returns `self` and does not mutate `X` / `y`.
- `coef_` has shape `(n_features,)`; `intercept_` is a Python `float`.
- The intercept column is never regularised.

## 4. Pitfalls

- Forgetting to centre before adding `α I` accidentally shrinks the bias.
- Very large `α` drives all weights to ~0; check scaling of features first.
