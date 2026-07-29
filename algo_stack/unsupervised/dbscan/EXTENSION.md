# DBSCAN — Extension Guide

## 1. Extension surface

- `metric` (euclidean / manhattan).
- Neighbourhood construction (swap brute force for a KD-tree).

## 2. Variants

- HDBSCAN / OPTICS for variable density.
- Weighted `min_samples`.

## 3. Invariants

- Noise label is always `-1`.
- Every index in `core_sample_indices_` has ≥ `min_samples` neighbours in `eps`.

## 4. Pitfalls

- `eps` is scale-sensitive — standardise features first.
- Very large `min_samples` can mark everything as noise.
