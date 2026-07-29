# L-BFGS — Extension Guide

## 1. Extension surface

- `minimize_lbfgs` kwargs: `m`, `tol`, `c1`, `c2`, `max_iter`.
- `wolfe_line_search` — swap for More–Thuente or backtracking Armijo.
- Two-loop recursion — add damping or compact representation.

## 2. Common variants

### 2.1 Owl-QN (L1-regularised)

Proximal L-BFGS for objectives `f + λ||x||₁`. Orthant-wise projected updates.

### 2.2 L-BFGS-B

Bound constraints `l ≤ x ≤ u`. Projected line search + Cauchy point
(Byrd et al.).

### 2.3 Online / stochastic L-BFGS

Accumulate `(s, y)` from mini-batch gradients with careful scaling
(SQN, oLBFGS).

### 2.4 Plug into estimators

`SoftmaxClassifier(optimizer="lbfgs")` already demonstrates wiring: flatten
parameters → minimise cross-entropy → unflatten.

## 3. Invariants

- Search direction must satisfy `p · ∇f < 0` (descent).
- History pairs must have `yᵀs > 0`.
- `fun` and `jac` must be consistent (finite-difference check in tests).

## 4. Pitfalls

- Noisy / mini-batch gradients break the secant equation — use full-batch
  objectives only.
- Very flat regions can make `yᵀy ≈ 0`; clamp `γ`.
- Line search may fail if `fun` is non-smooth; fall back to backtracking.

## 5. Suggested follow-up

- L-BFGS-B with box constraints
- Conjugate Gradient
- Trust-region Newton
