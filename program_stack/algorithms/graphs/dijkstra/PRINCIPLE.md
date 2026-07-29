# Dijkstra — Principle

## 1. Problem statement

Given a directed graph with **non-negative** edge weights, a node count `n`,
and a source `s`, compute `dist[v]` = length of a shortest `s → v` path
(or `INF` if unreachable).

## 2. Idea & invariants

- Maintain tentative distances; initially `dist[s] = 0`, others `INF`.
- Repeatedly select the unsettled node `u` with smallest `dist[u]` and
  **relax** all outgoing edges `u → v`.
- Invariant: once a node is settled, its distance is final (requires `w ≥ 0`).

## 3. Correctness sketch

When `u` is settled, any path to `u` that still uses unsettled nodes would
have length at least `dist[u]` because remaining edges are non-negative and
every unsettled node has tentative distance `≥ dist[u]`.

## 4. Complexity

| Quantity | Cost |
| -------- | ---- |
| Time (dense / O(V²) scan) | `O(V² + E)` ≈ `O(V²)` |
| Time (binary heap) | `O((V + E) log V)` |
| Extra space | `O(V + E)` |

## 5. Implementation notes

- Negative weights break the algorithm — use Bellman–Ford instead.
- This demo’s `INF` must be large enough that `INF + w` does not wrap for
  typical didactic inputs (languages check before adding).
