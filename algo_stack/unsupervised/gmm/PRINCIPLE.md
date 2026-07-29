# Gaussian Mixture (EM) — Principle

## 1. Problem statement

Model `p(x) = Σ_k π_k N(x | μ_k, Σ_k)` and maximise the incomplete-data likelihood.

## 2. EM

**E-step**: `γ_{ik} ∝ π_k N(x_i | μ_k, Σ_k)`.
**M-step**: weighted means / covariances / mixing weights from `γ`.

## 3. Algorithm

```
init means from random rows; equal weights; shared empirical cov
repeat: E-step responsibilities; M-step params
until |Δ mean loglik| < tol or max_iter
```

## 4. Complexity

`O(n · k · d²)` per iteration for full cov; `O(n · k · d)` for diag.

## 5. Notes

`reg_covar` floors eigenvalues / diagonal entries. Log densities use Cholesky for full cov.
