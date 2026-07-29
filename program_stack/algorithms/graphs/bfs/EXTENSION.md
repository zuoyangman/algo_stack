# BFS — Extension Guide

## 1. Extension surface

Safe to extend: graph representation helpers, optional distance / parent arrays,
multi-source BFS wrappers. Keep `bfs(adj, start) → visit order` as the core.

## 2. Common variants

### 2.1 Shortest paths (unweighted)
Record `dist[v]` / `parent[v]` when enqueueing; reconstruct path by walking parents.

### 2.2 Multi-source BFS
Enqueue all sources at distance 0 (e.g. nearest wall / infection spread).

### 2.3 0-1 BFS
Deque + edges of weight 0 or 1; not covered by plain FIFO BFS.

## 3. Invariants to preserve

- `main` exits 0 only on success.
- Success line: `bfs: ok`
- Cross-language API naming stays mappable.

## 4. Pitfalls

- Marking visited only on dequeue can enqueue the same node many times.
- Forgetting to bound neighbor indices causes out-of-range access.
- Directed vs undirected: missing reverse edges changes reachability.

## 5. Suggested follow-up algorithms

- DFS, Dijkstra, 0-1 BFS, bipartite check via 2-coloring BFS
