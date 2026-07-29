# Agglomerative Clustering — Extension Guide

## 1. Extension surface

- `linkage`, `n_clusters`.
- Distance matrix construction.

## 2. Variants

- Precomputed affinity matrix.
- NN-chain / MST acceleration for single linkage.

## 3. Invariants

- `children_` has `n_samples - n_clusters` rows when stopped early… actually
  we stop when `len(active) == n_clusters`, so merges = `n - n_clusters`.
- Labels are in `{0, …, n_clusters-1}`.

## 4. Pitfalls

- Single linkage chains; complete linkage prefers compact blobs.
- Ward with non-Euclidean metrics is invalid — do not add casually.
