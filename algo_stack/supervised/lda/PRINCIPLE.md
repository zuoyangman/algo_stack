# Linear Discriminant Analysis — Principle

## 1. Problem statement

LDA is a generative classifier: model `p(x | y = k) = N(μ_k, Σ)` with a
**shared** covariance `Σ`, then classify by maximising the posterior
`p(y = k | x) ∝ p(x | k) p(k)`.

## 2. Discriminant function

Taking the log posterior and dropping terms independent of `k`:

```
δ_k(x) = xᵀ Σ⁻¹ μ_k  −  ½ μ_kᵀ Σ⁻¹ μ_k  +  log π_k.
```

This is linear in `x`. Equivalently `δ(x) = X coefᵀ + intercept` with
`coef_k = Σ⁻¹ μ_k` and `intercept_k = −½ μ_kᵀ Σ⁻¹ μ_k + log π_k`.

## 3. Parameter estimates

```
μ̂_k  = mean of samples in class k
π̂_k  = n_k / n          (or user-supplied priors)
Σ̂    = (1/(n−K)) Σ_k Σ_{i:y_i=k} (x_i − μ̂_k)(x_i − μ̂_k)ᵀ
```

Optional shrinkage: `Σ ← (1−γ) Σ + γ diag(Σ)`.

## 4. Algorithm

```
1. Group samples by class; compute means_ and counts.
2. Accumulate pooled within-class scatter → cov_.
3. Apply shrinkage / ridge if requested.
4. priors_ ← counts/n or user priors.
5. coef_ ← solve(cov, means.T).T
6. intercept_ ← −½ sum(means * coef, axis=1) + log(priors)
```

## 5. Complexity

| Quantity | Cost |
| -------- | ---- |
| Training | `O(n d² + d³)` |
| Prediction | `O(K d)` per sample |

## 6. Notes

- When `Σ` differs per class → Quadratic Discriminant Analysis (QDA).
- LDA is equivalent to least-squares classification under balanced two-class
  Gaussian assumptions.
