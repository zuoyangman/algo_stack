# UMAP — Extension Guide

## 1. Extension surface

- Constructor: `n_neighbors`, `min_dist`, `n_epochs`, `learning_rate`, `init`.
- `_fuzzy_simplicial_set` — swap distance / membership construction.
- `_optimize_layout` — change attraction / repulsion schedule.
- `_fit_ab` — replace the `(a, b)` curve fit.

## 2. Variants

- **Random init only** — set `init="random"` for faster / noisier starts.
- **Larger `n`** — replace dense distances with approximate knn; store `B` as
  sparse edges only.
- **Out-of-sample** — embed new points by optimising only their coordinates
  against fixed training `embedding_` (not implemented here).
- **Supervised UMAP** — blend label-aware distances into the fuzzy graph.

## 3. Invariants

- `n_neighbors >= 2` and `n_samples >= 3`.
- `embedding_.shape == (n_samples, n_components)`.
- Fuzzy diagonal is zero; memberships in `[0, 1]`.

## 4. Pitfalls

- Too few epochs → clusters may not separate.
- `n_neighbors` close to `n_samples` flattens local structure.
- Without `random_state`, negative sampling makes runs non-reproducible.

## 5. Suggested follow-up

- Sparse / approximate UMAP for large `n`
- Parametric UMAP (neural encoder)
- Compare with `algo_stack.unsupervised.tsne`
