# Union–Find — Principle

## 1. Problem statement

Maintain a partition of `{0, …, n-1}` under merge (union) and same-component
queries (connected / find).

## 2. Idea & invariants

Each set is a rooted tree. `parent[i]` points toward the root; a root has
`parent[r] == r`. `rank[r]` is an upper bound on tree height used to attach the
shallower tree under the deeper one (union by rank).

Path compression: during `find`, point nodes directly at the root.

## 3. Correctness sketch

- `find` follows parents to a root; compression does not change which root a
  node belongs to.
- `union` links two distinct roots; afterward both sides share one root, so
  `connected` becomes true.
- Elements start as singleton roots — the partition invariant holds initially
  and is preserved by union.

## 4. Complexity

| Quantity | Cost |
| -------- | ---- |
| `find` / `union` / `connected` (amortized) | Nearly `O(1)` — `O(α(n))` |
| Extra space | `O(n)` |

`α` is the inverse Ackermann function (effectively ≤ 4 for practical `n`).

## 5. Implementation notes

Initialize `parent[i] = i`, `rank[i] = 0`. Prefer iterative find with two-pass
or one-pass compression. Union by size is an equivalent alternative to rank.
