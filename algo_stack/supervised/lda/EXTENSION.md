# Linear Discriminant Analysis — Extension Guide

## 1. Extension surface

- **`priors` / `shrinkage`**: already exposed.
- **`solver`**: swap `np.linalg.solve` for SVD / eigen-decomposition of
  `S_w⁻¹ S_b` to also expose a dimensionality-reducing transform.
- **`n_components`**: keep only the top `K−1` discriminant directions for
  visualisation / feature extraction (`transform` method).

## 2. Common variants

### 2.1 Quadratic Discriminant Analysis

Estimate a separate `cov_k` per class; decision becomes quadratic.
Store `covariances_` list instead of a single `cov_`.

### 2.2 Regularised / shrinkage LDA

Ledoit–Wolf or OAS estimators for `Σ` when `d > n`.

### 2.3 Sparse LDA / Penalised LDA

Add an L1 penalty on `coef_` and solve via proximal methods.

## 3. Invariants

- `coef_.shape == (n_classes, n_features)`
- `intercept_.shape == (n_classes,)`
- `predict` returns labels from `classes_`

## 4. Pitfalls

- Singular `cov_` when `d ≥ n − K` — use shrinkage or a small diagonal ridge.
- Highly imbalanced classes: supply explicit `priors` if the training
  frequencies do not match deployment.
