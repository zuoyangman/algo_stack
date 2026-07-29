# DBSCAN — Principle

## 1. Problem statement

Partition points into dense regions separated by sparse areas; outliers are noise.

## 2. Model

A point is **core** if it has ≥ `min_samples` neighbours within distance `eps`.
Clusters are maximal sets of density-reachable points from core seeds.

## 3. Algorithm

```
compute pairwise distances
mark core points
for each unlabelled core point:
  BFS through neighbours of core points → new cluster id
non-core points claimed only when reached from a core; else noise (-1)
```

## 4. Complexity

Brute-force `O(n²)` time and memory (toy / didactic).

## 5. Notes

Self is counted in the neighbourhood (standard DBSCAN).
