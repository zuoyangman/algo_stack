# StandardScaler — Principle

## 1. Problem statement

Features on different scales dominate distance- and gradient-based learners.
Standardisation maps each column to (approximately) zero mean and unit variance.

## 2. Model

For feature `j`:

```
z_{ij} = (x_{ij} − μ_j) / σ_j
```

with `μ_j = mean(X[:, j])` and `σ_j = std(X[:, j])` (population std, `ddof=0`).
If `σ_j = 0`, use `σ_j = 1` to avoid division by zero.

## 3. Algorithm

```
μ ← mean(X, axis=0)   if with_mean else 0
σ ← std(X, axis=0)    if with_std  else 1
σ[σ == 0] ← 1
transform: (X − μ) / σ
inverse:   Z * σ + μ
```

## 4. Complexity

`O(n d)` time and `O(d)` extra memory for the fitted statistics.

## 5. Notes

This module mirrors the utility in `algo_stack.utils.preprocessing.StandardScaler`
but inherits `BaseEstimator` + `TransformerMixin` and lives under the
`preprocessing` category package.
