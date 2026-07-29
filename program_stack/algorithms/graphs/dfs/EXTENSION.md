# DFS — Extension Guide

## 1. Extension surface

Core: `dfs(adj, start) → visit order`. Safe extensions: timestamps
(discovery/finish), parent edges, cycle detection, connected components.

## 2. Common variants

### 2.1 Iterative DFS
Push neighbors onto an explicit stack (reverse neighbor order to mimic
recursive preorder).

### 2.2 Full forest DFS
Loop over all nodes and start DFS from each unvisited node (components / topo).

### 2.3 Path finding / backtracking
Keep a path vector; pop on return for enumeration problems.

## 3. Invariants to preserve

- `main` exits 0 only on success.
- Success line: `dfs: ok`
- Cross-language API naming stays mappable.

## 4. Pitfalls

- Not marking visited before recursing can infinite-loop on cycles.
- Neighbor list order changes the reported visit sequence (tests fix an order).

## 5. Suggested follow-up algorithms

- Topological sort, Tarjan SCC, bridge/articulation-point finding
