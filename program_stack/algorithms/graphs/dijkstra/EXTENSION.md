# Dijkstra — Extension Guide

## 1. Extension surface

Core: `dijkstra(n, edges, source) → dist[]`. Safe to add: parent array for path
reconstruction, early exit when target is settled, adjacency-list builders.

## 2. Common variants

### 2.1 Binary / Fibonacci heap
Replace the O(V) min-scan with a priority queue for sparse graphs.

### 2.2 Undirected graphs
Insert both `(u,v,w)` and `(v,u,w)`.

### 2.3 Counting shortest paths
Extra DP on DAG of shortest-path edges after distances are known.

## 3. Invariants to preserve

- `main` exits 0 only on success.
- Success line: `dijkstra: ok`
- Cross-language API naming stays mappable.

## 4. Pitfalls

- Integer overflow when adding weights to `INF`.
- Using Dijkstra with negative edges yields wrong answers silently.
- Forgetting to skip already-settled nodes wastes work (still correct for O(V²)).

## 5. Suggested follow-up algorithms

- Bellman–Ford, Floyd–Warshall, A*, Prim’s MST
