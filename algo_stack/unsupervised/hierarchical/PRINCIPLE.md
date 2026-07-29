# Agglomerative Clustering — Principle

## 1. Problem statement

Start with `n` singletons; repeatedly merge the closest pair of clusters until
`n_clusters` remain.

## 2. Linkages

- single: min pairwise distance
- complete: max
- average: size-weighted mean of distances
- ward: Lance–Williams update on *squared* Euclidean distances

## 3. Algorithm

Naive scan for the closest active pair each step → `O(n³)`.

## 4. Complexity

| Quantity | Cost |
| -------- | ---- |
| Time | `O(n³)` |
| Memory | `O(n²)` for distances |

## 5. Notes

Ward assumes Euclidean geometry (implemented via squared distances + LW formula).
