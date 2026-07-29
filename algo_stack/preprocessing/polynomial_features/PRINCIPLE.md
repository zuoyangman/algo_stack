# PolynomialFeatures — Principle

## 1. Problem statement

Linear models only fit hyperplanes in the original features. Mapping `x ↦ φ(x)`
into a polynomial basis lets them fit curved decision / regression surfaces.

## 2. Model

Given `x ∈ ℝ^d` and degree `p`, emit all monomials

```
∏_{j=1}^{d} x_j^{α_j}    with  |α|_1 ≤ p
```

- `include_bias=True` keeps the empty product `1` (`|α|_1 = 0`).
- `interaction_only=True` keeps only multi-indices with entries in `{0, 1}`
  (no pure powers).

## 3. Algorithm

```
enumerate combinations (with/without replacement) of feature indices
  for degrees start..degree
powers_[i, j] ← count of feature j in combination i
XP[:, i] ← ∏_j X[:, j] ** powers_[i, j]
```

## 4. Complexity

Output width is `C(d + p, p)` (or smaller with `interaction_only`);
transform is `O(n · n_out · d)` naïvely.

## 5. Notes

`powers_` makes the expansion inspectable and is the single source of truth
for both fit-time sizing and transform-time evaluation.
