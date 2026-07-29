# Lasso Regression — Extension Guide

## 1. Extension surface

- **`alpha` / `tol` / `max_iter`**: standard CD hyperparameters.
- **Warm start**: initialise `w` from a previous fit (useful for α-paths).
- **Screening rules**: skip coordinates proven to stay at zero (SAFE / strong
  rules) for large-`d` problems.

## 2. Common variants

### 2.1 Elastic Net

Add an L2 term: objective becomes
`(1/(2n))||·||² + α ρ ||w||₁ + α (1−ρ)/2 ||w||²`. The CD update gains an
extra `(1 + n α (1−ρ))` factor in the denominator.

### 2.2 Weighted / group Lasso

Replace the scalar soft-threshold with a group soft-threshold
`w_g ← (1 − γ / ||ρ_g||)_+ ρ_g` for feature groups.

### 2.3 Regularisation path

Fit a decreasing sequence of `α` values, warm-starting each solve from the
previous solution (LAR / pathwise CD).

## 3. Invariants

- `fit` returns `self`; `X` / `y` are not mutated in place.
- `coef_` shape `(n_features,)`; many entries may be exactly `0.0`.
- Intercept is never L1-penalised.

## 4. Pitfalls

- Too large `α` zeros every weight — lower it or rescale `y`.
- Without feature scaling, large-norm columns dominate selection.
- Coordinate descent is sensitive to feature correlations; Elastic Net is
  often more stable.
