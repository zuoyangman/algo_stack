# K-Means — Principle

## 1. Problem statement

Given `n` samples `X ∈ ℝ^{n×d}` and an integer `k`, partition the data into
`k` clusters `{C_1, …, C_k}` and find centroids `μ_1, …, μ_k ∈ ℝᵈ` that
minimise the within-cluster sum of squared distances (a.k.a. *inertia*):

```
J(C, μ) = Σ_{j=1..k} Σ_{x ∈ C_j} ||x − μ_j||².
```

This is NP-hard in general; Lloyd's algorithm is a fast local optimiser.

## 2. Mathematical model

Two coupled minimisation problems:

1. **Assignment**: given centroids `μ`, the optimal partition is
   `c(x) = argmin_j ||x − μ_j||²`.
2. **Update**: given partition `C`, the optimal centroid is
   `μ_j = mean(C_j)`.

Both sub-problems are convex; alternating between them decreases `J`
monotonically, so the algorithm converges to a local minimum in finite
steps (the number of distinct partitions is finite).

## 3. Derivation

The centroid update is the closed-form solution of
`min_μ Σ_{x ∈ C_j} ||x − μ||²`, whose gradient is
`2 Σ (μ − x) = 0  ⇒  μ = mean(C_j)`.

The assignment step is by definition the nearest-neighbour rule.

## 4. Algorithm (Lloyd + k-means++)

```
init:
  k-means++: pick centre c1 uniformly; then iteratively sample new centres
             with probability ∝ D(x)² where D(x) is the squared distance
             from x to its currently-closest centre.

repeat (up to max_iter times):
  d² = squared distances X ↔ centres
  labels = argmin over centres
  for each cluster:
     if non-empty: centre ← mean(points)
     else:         centre ← farthest point (avoid empty cluster)
  if ||new − old|| / ||old|| < tol: break
```

We run the whole thing `n_init` times with different RNG seeds and keep the
solution with the lowest inertia. This is the standard way to mitigate the
sensitivity of Lloyd's algorithm to initialisation.

## 5. Complexity

| Quantity | Cost |
| -------- | ---- |
| Init (k-means++) | `O(n · k · d)` |
| Per iteration    | `O(n · k · d)` (one matrix multiply) |
| Memory           | `O(n · k)` for the distance matrix |

## 6. Numerical & implementation notes

- Distances are computed via the `||x||² + ||μ||² − 2 x·μ` identity for a
  single matrix multiplication; `np.maximum(d², 0)` guards against tiny
  negative results from float cancellation.
- Empty clusters are handled by re-seeding to the point farthest from any
  current centroid. Naïve implementations crash on `X[mask].mean(axis=0)`
  when `mask` is empty.
- Convergence is checked on **relative** Frobenius-norm centroid shift,
  which makes the `tol` value scale-independent (relative to the magnitude
  of the centroids).
- We seed each `n_init` run from a fresh `Generator` derived from the master
  RNG, so all `n_init` restarts are reproducible from a single
  `random_state`.
