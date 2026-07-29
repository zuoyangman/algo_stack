# Topological Sort — Extension Guide

## 1. Extension surface

Core: `topologicalSort(adj) → order or failure`. Safe to add: edge-list input,
lexicographically smallest order (min-heap queue), cycle edge reporting.

## 2. Common variants

### 2.1 DFS finish-time reverse
Run forest DFS; prepend node on finish. Grey-node back edge ⇒ cycle.

### 2.2 Lexicographic order
Use a priority queue of indegree-0 nodes instead of a plain FIFO.

### 2.3 Partial order queries
Build topo order then answer “must A precede B?” via positions / reachability.

## 3. Invariants to preserve

- `main` exits 0 only on success.
- Success line: `topological_sort: ok`
- Cross-language API naming stays mappable.

## 4. Pitfalls

- Treating a cyclic graph as sorted if you forget to check `|order| == V`.
- Mutating indegrees in place when the caller reuses the graph later.

## 5. Suggested follow-up algorithms

- Course-schedule style problems, critical path on DAG, Tarjan SCC
