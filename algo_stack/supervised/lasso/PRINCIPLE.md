# Lasso Regression — Principle

## 1. Problem statement

Lasso replaces Ridge's L2 penalty with an L1 penalty. The non-differentiable
`||w||₁` term drives many coordinates exactly to zero, yielding sparse
models that perform automatic feature selection.

## 2. Mathematical model

```
L(w, b) = (1 / (2 n)) ||X w + b − y||²  +  α ||w||₁,   α ≥ 0.
```

The intercept is not penalised. After centring, we optimise over `w` only.

## 3. Soft-thresholding

The proximal operator of `α |·|` is the soft-threshold:

```
S(z, γ) = sign(z) · max(|z| − γ, 0).
```

For a single coordinate `j`, holding others fixed, the minimiser is

```
w_j* = S( X_jᵀ r_j , n α ) / ||X_j||²
```

where `r_j = y − X w + w_j X_j` is the partial residual.

## 4. Algorithm (cyclic coordinate descent)

```
1. Centre X, y if fit_intercept.
2. w ← 0, residual ← y.
3. For iter = 1 … max_iter:
     For j = 1 … d:
       ρ ← X_jᵀ residual + w_j ||X_j||²
       w_j' ← S(ρ, n α) / ||X_j||²
       Update residual by the change in w_j.
     Stop if max |Δw_j| < tol.
4. Recover intercept from means.
```

## 5. Complexity

| Quantity | Cost |
| -------- | ---- |
| Per sweep | `O(n d)` |
| Worst case | `O(max_iter · n d)` |

## 6. Numerical notes

- Column norms are precomputed once; constant columns are guarded against
  zero division.
- Larger `α` → sparser `coef_`.
- Scale features before fitting so the L1 ball treats coordinates fairly.
