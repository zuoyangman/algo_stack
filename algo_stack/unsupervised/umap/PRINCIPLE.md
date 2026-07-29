# UMAP — Principle

## 1. Problem statement

Embed high-dimensional points in a low-d space so that **local fuzzy
neighbourhoods** are preserved. UMAP builds a weighted knn graph in the
original space, then lays out points by matching a low-d fuzzy graph via
cross-entropy.

## 2. Model (simplified)

**High-d fuzzy simplicial set**

1. For each point `i`, take its `k = n_neighbors` nearest neighbours.
2. Let `ρ_i` be the distance to the nearest neighbour.
3. Binary-search a local scale `σ_i` so
   `Σ_{j ∈ knn(i)} exp(−max(0, d_ij − ρ_i) / σ_i) ≈ log₂(k)`.
4. Directed memberships `μ_{i|j}`; symmetrise with fuzzy set union
   `B_ij = μ_ij + μ_ji − μ_ij μ_ji`.

**Low-d similarities**

`v_ij = 1 / (1 + a ‖y_i − y_j‖^{2b})`, with `(a, b)` fit so the curve
respects `min_dist`.

**Objective**

Minimise fuzzy cross-entropy between `B` and `v` with SGD: attraction along
positive edges, repulsion via negative sampling.

## 3. Algorithm

```
build knn + local σ via binary search; form B
fit (a, b) from min_dist
init Y via Laplacian eigenmaps (or random)
for epoch = 1..n_epochs:
  for each edge (i, j) with weight w:
    attract i, j according to ∂CE/∂y
    for n_neg random k: repel i from k
  centre Y
```

## 4. Complexity

Dense distances `O(n² d)` once; layout `O(n_epochs · |E| · n_neg · n_components)`.
Intended for **`n ≲ 100`** in this didactic build (no approximate knn / Barnes–Hut).

## 5. Design simplifications

Documented departures from production UMAP (`umap-learn`):

- Dense all-pairs distances (no NN-descent / RP trees).
- Dense membership matrix (no sparse CSR graph).
- Fixed negative-sample count; no edge-epoch scheduling table.
- No PCA preprocessing or angular metrics.
- `transform` does **not** embed new points — returns `embedding_` when
  `n_samples` matches the training set, else raises `NotImplementedError`.

## 6. Notes

Spectral init uses the normalised Laplacian of `B`; falls back to small
Gaussian noise if eigendecomposition is insufficient.
