# PolynomialFeatures — Extension Guide

## 1. Extension surface

- `degree`, `include_bias`, `interaction_only`.
- `powers_` can drive custom bases (e.g. only selected interactions).

## 2. Variants

- Spline / piecewise polynomial bases.
- Orthogonal polynomials for better conditioning.
- Sparse interaction selection (forward stepwise).

## 3. Invariants

- Column 0 is all-ones when `include_bias=True`.
- `transform` output shape is `(n_samples, n_output_features_)`.

## 4. Pitfalls

- Combinatorial explosion: `degree=3`, `d=20` is already huge.
- High powers amplify scale differences — pair with `StandardScaler` first.
