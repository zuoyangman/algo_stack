# MinMaxScaler — Extension Guide

## 1. Extension surface

- `feature_range` — any `(min, max)` with `min < max`.
- Could clip out-of-range values at transform time.

## 2. Variants

- `MaxAbsScaler` — divide by `max |x|` without shifting.
- Robust min–max using percentiles.

## 3. Invariants

- After fit on training data, each training column lies in `feature_range`.
- `inverse_transform(transform(X)) ≈ X` when ranges are non-zero.

## 4. Pitfalls

- Sensitive to outliers (they stretch `data_min_` / `data_max_`).
- Test values outside the training range produce outputs outside `feature_range`.
