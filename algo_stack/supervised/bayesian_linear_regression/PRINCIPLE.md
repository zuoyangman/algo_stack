# Bayesian Linear Regression — Principle

## 1. Generative model

```
p(w)          = N(w | 0, α⁻¹ I)          # isotropic Gaussian prior
p(y | X, w)   = N(y | X w, β⁻¹ I)         # Gaussian likelihood
```

`α` = prior precision, `β` = noise precision. The intercept is handled by
centring (empirical Bayes mean) and is not given a prior here.

## 2. Posterior

Because prior and likelihood are conjugate Gaussians, the posterior is

```
p(w | X, y) = N(w | m_N, S_N)

S_N⁻¹ = α I + β Xᵀ X
m_N   = β S_N Xᵀ y
```

`coef_ = m_N`, `sigma_ = S_N`.

## 3. Predictive distribution

For a new input `x*`:

```
p(y* | x*, X, y) = N(y* | m_Nᵀ x*,  σ²(x*))
σ²(x*) = β⁻¹ + x*ᵀ S_N x*
```

`predict` returns the mean; `predict_std` returns `√σ²(x*)`.

## 4. Algorithm

```
1. Centre X, y if fit_intercept.
2. A ← β XᵀX + α I
3. sigma_ ← A⁻¹
4. coef_  ← β sigma_ Xᵀ y
5. intercept_ ← ȳ − x̄ᵀ coef_
```

## 5. Complexity

| Quantity | Cost |
| -------- | ---- |
| Training | `O(n d² + d³)` |
| `predict` / `predict_std` | `O(d)` / `O(d²)` per sample |

## 6. Notes

- Larger `α` → stronger shrinkage (like Ridge with `λ = α/β`).
- Larger `β` → trust the data more → smaller predictive variance.
- Evidence maximisation can learn `(α, β)` from data (see EXTENSION).
