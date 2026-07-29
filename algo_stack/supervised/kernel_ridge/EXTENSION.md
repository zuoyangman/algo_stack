# Kernel Ridge — Extension Guide

## 1. Extension surface

- **Kernels**: extend `algo_stack.utils.kernels.kernel_matrix` (sigmoid,
  Laplacian, custom callables).
- **Solvers**: swap `np.linalg.solve` for conjugate gradients / Nyström /
  incomplete Cholesky when `n` is large.
- **Multi-output**: solve `(K + λI) A = Y` for matrix `Y` (same `K`).

## 2. Common variants

### 2.1 Nyström approximation

Sample `m ≪ n` landmark points, form `K_{nm}` and `K_{mm}`, approximate
`K ≈ K_{nm} K_{mm}^{-1} K_{nm}^T`, then solve in the `m`-dimensional
feature space — `O(n m²)` instead of `O(n³)`.

### 2.2 Random Fourier features

For shift-invariant kernels (RBF), approximate `φ(x)` with Monte-Carlo
features and run **primal** ridge regression in `ℝ^D`.

### 2.3 Kernel logistic / SVM

Keep the same `K`, change the loss (logistic, hinge). The closed form is
lost; use iterative dual / primal solvers.

## 3. Invariants

- `dual_coef_` has length `n_train` and `predict` always uses `X_fit_`.
- `gamma_` is frozen at fit time.
- `fit` does not modify caller arrays.

## 4. Pitfalls

- `alpha=0` with a singular `K` (e.g. duplicate rows + linear kernel) can
  make the solve fail — keep `alpha > 0` in practice.
- Polynomial kernels with large `degree` / `gamma` overflow easily.
- Do not standardise inside the estimator; scale features upstream if needed.

## 5. Suggested follow-ups

- Gaussian Process Regression (same `K`, probabilistic predictive variance)
- Multiple Kernel Learning
- Sparse KRR / subset of regressors
