# Bayesian Linear Regression — Extension Guide

## 1. Extension surface

- **`alpha` / `beta`**: fixed precisions. Replace with evidence maximisation
  (type-II ML) that iterates

  ```
  γ = Σ_i λ_i / (α + λ_i)          # λ_i = eigenvalues of β XᵀX
  α ← γ / ||m_N||²
  β ← (n − γ) / ||y − X m_N||²
  ```

  until convergence; store the learned values on `alpha_` / `beta_`.

- **`sigma_`**: already exposed for uncertainty propagation.

## 2. Common variants

### 2.1 Automatic Relevance Determination (ARD)

Use a diagonal prior `diag(α_1 … α_d)` and prune features whose `α_j → ∞`.

### 2.2 Bayesian Ridge (sklearn-style)

Jointly estimate `α, β` via the evidence procedure above — expose as
`BayesianRidge` sibling or a `fit_alpha_beta=True` flag.

### 2.3 Basis expansions

Apply polynomial / RBF features first; the same posterior formulae hold with
`φ(X)` replacing `X`.

## 3. Invariants

- `predict` returns the posterior predictive **mean** only.
- `predict_std(X)` shape matches `(n_samples,)`.
- `alpha_`, `beta_` are positive floats stored after `fit`.

## 4. Pitfalls

- Explicit `inv(A)` can be unstable; prefer `solve` + a cached Cholesky factor
  for production code.
- Setting `beta` far from the true noise level miscalibrates `predict_std`.
