# Gaussian Mixture (EM) — Extension Guide

## 1. Extension surface

- `covariance_type`, `reg_covar`, `n_components`.
- `_e_step` / `_m_step` hooks.

## 2. Variants

- Tied / spherical covariances.
- Multiple random restarts (keep highest `lower_bound_`).

## 3. Invariants

- `weights_` sum to 1; covariances stay PD via `reg_covar`.
- `predict` = argmax of `predict_proba`.

## 4. Pitfalls

- Singular covariances without `reg_covar` on collinear data.
- Label switching across random seeds.
