# PCA — Extension Guide

## 1. Extension surface

- `n_components` (could accept variance fraction later).
- SVD backend.

## 2. Variants

- Whitening (divide by `√explained_variance_`).
- Incremental / randomised SVD for large `n`.

## 3. Invariants

- `components_` rows are orthonormal (up to numerics).
- `explained_variance_ratio_` sums to ≤ 1.

## 4. Pitfalls

- Always centre — skipping breaks PCA.
- More components than rank yields a ValueError.
