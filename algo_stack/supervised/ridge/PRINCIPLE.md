# Ridge Regression — Principle

## 1. Problem statement

OLS becomes unstable when features are correlated or when `d ≈ n`. Ridge
adds an L2 penalty that shrinks weights toward zero and guarantees a unique
solution even when `XᵀX` is singular.

## 2. Mathematical model

```
L(w, b) = ||X w + b − y||²  +  α ||w||²,   α ≥ 0.
```

The intercept `b` is **not** included in the penalty.

## 3. Derivation

Centre `X` and `y` so the intercept drops out. Setting `∇w L = 0`:

```
2 Xcᵀ (Xc w − yc) + 2 α w = 0
  ⇒  (Xcᵀ Xc + α I) w* = Xcᵀ yc.
```

Recover `b* = ȳ − x̄ᵀ w*`. When `α = 0` this reduces to OLS.

## 4. Algorithm

```
1. Validate inputs.
2. If fit_intercept: centre X and y; else leave as-is.
3. Solve (XᵀX + α I) w = Xᵀ y with np.linalg.solve.
4. intercept_ = y_mean − X_mean @ w  (or 0).
```

## 5. Complexity

| Quantity | Cost |
| -------- | ---- |
| Training time   | `O(n d² + d³)` |
| Prediction time | `O(d)` per sample. |

## 6. Numerical notes

- Prefer `solve` over an explicit inverse.
- Larger `α` → stronger shrinkage → smaller `||coef_||`.
- Features should usually be scaled (e.g. `StandardScaler`) so the penalty
  treats all coordinates fairly.
