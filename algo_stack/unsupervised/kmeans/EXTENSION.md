# K-Means — Extension Guide

## 1. Extension surface

- **Constructor**: `n_clusters`, `init`, `n_init`, `max_iter`, `tol`,
  `random_state`. New hyperparameters belong here; preserve keyword-only.
- **`_init_centres`**: dedicated method for the initialisation strategy.
  Add new strategies by extending the `if/elif` chain.
- **`_single_run`**: implements one Lloyd run. Subclasses or variants should
  override this rather than `fit`, so that the `n_init` restart logic in
  `fit` keeps working.
- **`_sq_dist_to_centres`**: free function — swap it out for variants that
  need a different distance (kernel k-means, etc.).

## 2. Common variants and how to add them

### 2.1 Mini-batch K-Means

Replace `_single_run` with a loop that, at each step, samples
`batch_size` indices instead of using all of `X`, and updates centroids
with a running mean (per-cluster sample count maintained on the side).
Add `batch_size: int` to `__init__`.

### 2.2 K-Medoids (PAM)

Centroids must be **actual data points**, and the update minimises a sum of
distances rather than squared distances. The cleanest plug-in point is
`_single_run`: replace the `X[mask].mean(...)` line with a selection of the
medoid (the point in `C_j` with smallest sum of pairwise distances). Best
implemented as a sibling class `KMedoids` because the math is meaningfully
different.

### 2.3 Kernel K-Means

Distances in the assignment step are computed in feature space via a
kernel: `||φ(x) − μ_j||² = K(x, x) − 2 (1/|C_j|) Σ_{y ∈ C_j} K(x, y) +
(1/|C_j|²) Σ_{y, y' ∈ C_j} K(y, y')`. Replace `_sq_dist_to_centres` with a
kernel-aware function. Note that the `cluster_centers_` attribute then
lives in feature space and is no longer expressible in input space.

### 2.4 Constrained / weighted K-Means

Pass `sample_weight: np.ndarray | None = None` to `fit`. Multiply each
point's contribution to its centroid by its weight; the centroid update
becomes the weighted mean.

## 3. Invariants the implementation relies on

- `fit` returns `self` and does not mutate `X`.
- After `fit`, `labels_.shape == (n_samples,)`,
  `cluster_centers_.shape == (n_clusters, n_features)`, and
  `inertia_ = Σ_i ||x_i − μ_{labels_[i]}||²` for the **best run**.
- Empty clusters are *not* allowed at convergence — `_single_run` handles
  them by re-seeding.
- The same `random_state` always produces the same `(labels_, centres_,
  inertia_)` triple.

## 4. Pitfalls

- Running with `n_init=1` is fragile: K-Means is notoriously sensitive to
  the initial seeding. Always default to `n_init ≥ 10` unless you have a
  good reason.
- Convergence on the relative shift can stall for clusters with very large
  scale differences in `X`; consider standardising features first.
- `inertia_` is **not** comparable across different scalings of `X` — never
  use it to pick `k` without also normalising the inputs.
- For very large `n` the `O(n × k)` distance matrix dominates memory; switch
  to mini-batch (§2.1) before adding any tree / index structure.

## 5. Suggested follow-up algorithms

- Mini-batch K-Means
- Gaussian Mixture Models (soft k-means via EM)
- DBSCAN
- Spectral Clustering
- Bisecting K-Means / Hierarchical clustering
