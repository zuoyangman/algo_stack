# Union–Find — Extension Guide

## 1. Extension surface

Arrays `parent` / `rank` (or `size`) and ops `find`, `union`, `connected`.
Optional: component count, set size of a root.

## 2. Common variants

### 2.1 Union by size
Attach smaller tree under larger; store subtree sizes at roots.

### 2.2 Rollback / persistent DSU
Avoid path compression; use explicit undo stacks for offline queries.

### 2.3 Weighted / bipartite DSU
Store relative relation (xor/diff) along parent edges.

## 3. Invariants to preserve

- `main` exits 0 only on success.
- Success line: `union_find: ok`
- Cross-language API naming stays mappable.

## 4. Pitfalls

- Calling `union` without finding roots first.
- Mixing compressed parents with outdated ranks (ranks only meaningful at roots).
- Out-of-range element indices.

## 5. Suggested follow-up algorithms

- Kruskal MST
- Connected components on grids
- Cycle detection in undirected graphs
