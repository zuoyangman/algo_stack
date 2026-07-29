# StandardScaler — Extension Guide

## 1. Extension surface

- `with_mean` / `with_std` — disable centring or scaling independently.
- Could add `copy=False` in-place transform later.

## 2. Variants

- RobustScaler (median / IQR) for outlier-heavy data.
- MaxAbsScaler for sparse-friendly scaling.

## 3. Invariants

- After `fit_transform` with both flags True on non-constant data,
  column means ≈ 0 and column stds ≈ 1.
- `inverse_transform(transform(X)) ≈ X`.

## 4. Pitfalls

- Fit only on training data; apply the same `mean_` / `scale_` to the test set.
- Constant columns get `scale_=1` — they remain unchanged after centring.
