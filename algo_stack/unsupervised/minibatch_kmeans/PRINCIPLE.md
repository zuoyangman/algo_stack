# Mini-Batch K-Means — Principle

## 1. Problem statement

Same objective as K-Means (minimise inertia), but each iteration updates
centroids using only a random subset of size `batch_size`.

## 2. Mathematical model

Per assigned point `x` to cluster `c` with visit count `v_c`:
`η = 1/v_c`, `μ_c ← (1−η) μ_c + η x` (online mean).

## 3. Algorithm

```
init centres (k-means++ / random)
counts ← 0
for t = 1..max_iter:
  sample batch B
  assign each x∈B to nearest centre
  for each assignment: update count + online mean
recompute full-data labels / inertia; keep best of n_init runs
```

## 4. Complexity

| Quantity | Cost |
| -------- | ---- |
| Per iteration | `O(batch_size · k · d)` |
| Memory | `O(k · d)` besides data |

## 5. Notes

Distances use `||x||²+||μ||²−2x·μ`. Empty clusters are re-seeded.
