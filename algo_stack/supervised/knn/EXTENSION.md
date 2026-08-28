# KNN — Extension Guide

## 1. Extension surface

- **Constructor**: `n_neighbors`, `weights`, `metric` — add more values
  (`"manhattan"`, `"chebyshev"`, `"minkowski(p=…)"`, callable) by extending
  `_validate_init` and `_pairwise_sq_euclidean` into a dispatcher
  (e.g. `_pairwise_distance(A, B, metric, **kw)`).
- **`_KNNBase._kneighbors`**: this is the single neighbour-lookup function
  shared between classifier and regressor. Replace its body to plug in a
  KD-tree, Ball-tree, or approximate-NN backend (FAISS / hnswlib).
- **Aggregation rules**: `predict_proba` (classifier) and `predict`
  (regressor) only consume the `(dists, indices)` pair returned by
  `_kneighbors`, so new neighbour backends compose cleanly.

## 2. Common variants and how to add them

### 2.1 New distance metric

```
def _pairwise_distance(A, B, metric, p=2):
    if metric == "euclidean":  return _pairwise_sq_euclidean(A, B) ** 0.5
    if metric == "manhattan":  return np.abs(A[:, None, :] - B[None, :, :]).sum(-1)
    if metric == "minkowski":  return (np.abs(A[:, None, :] - B[None, :, :]) ** p).sum(-1) ** (1/p)
    if callable(metric):       return metric(A, B)
    raise ValueError(...)
```

Be careful: only Euclidean admits the cheap `||a||² + ||b||² − 2 a·b`
trick. For other metrics you generally need a `O(n_q · n_t · d)`
broadcasting computation, which costs significantly more memory.

### 2.2 KD-tree / Ball-tree acceleration

Add `algorithm: str = "brute"` constructor argument; for
`algorithm in ("kdtree", "balltree")` build the tree in `fit` and use it in
`_kneighbors`. KD-trees only help for `d ≲ 20`; in higher dimensions the
brute-force matmul wins again.

### 2.3 Approximate nearest neighbours

Same dispatch as §2.2, but the backend is `hnswlib` / `faiss`. Be aware:
this introduces a non-NumPy dependency and breaks the "NumPy-only" rule;
keep the brute-force path as the default and the reference.

### 2.4 KNN density estimation / outlier detection

These are different *use-cases* for the same neighbour lookup. Expose
`kneighbors(X)` as a public method and add e.g. `LocalOutlierFactor` /
`KNNDensity` classes that consume it.

## 3. Invariants the implementation relies on

- `fit` does not modify `X` or `y`.
- `_kneighbors` always returns distances of shape `(n_query, k_effective)`
  sorted by ascending distance, where `k_effective = min(n_neighbors, n_train)`.
  Callers (classifier/regressor) rely on the sorted order only for the
  *distance-weighted* path; the uniform path is order-invariant.
- For the classifier, `y_train_` contains **integer class indices**, and
  `self.classes_[i]` is the actual label of index `i`.

## 4. Pitfalls

- Don't standardise inside the model — let the user pass a standardised
  `X` (or pipeline a `StandardScaler` outside). Distances are extremely
  scale-sensitive.
- For very large training sets, the `(n_q × n_t)` distance matrix can
  exceed memory. Add a `chunk_size` keyword and loop over query chunks
  before adding any tree backend.
- Ties in voting are broken by `np.argmax`'s deterministic-but-unspecified
  rule (returns the lowest index). If ties matter, prefer odd `k`.

## 5. Suggested follow-up algorithms

- Weighted / Kernel KNN
- Local Outlier Factor (LOF)
- KD-tree / Ball-tree based neighbour search as a stand-alone module
- Locality-sensitive hashing (LSH)
