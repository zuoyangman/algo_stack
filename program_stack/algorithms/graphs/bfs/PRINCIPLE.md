# BFS — Principle

## 1. Problem statement

Given a directed (or undirected) graph as an adjacency list and a start node
`s`, produce the order in which nodes are first discovered when exploring
breadth-first from `s`. Unreachable nodes are omitted.

## 2. Idea & invariants

- Maintain a FIFO queue of discovered-but-not-yet-expanded nodes.
- Mark a node visited when it is **enqueued** (not when dequeued) so each node
  enters the queue at most once.
- Invariant: every node in the queue is at distance `d` or `d+1` from `s`, and
  all nodes at distance `< d` have already been expanded.

## 3. Correctness sketch

Nodes are discovered in non-decreasing distance from `s`. The first time a
node is reached is along a shortest path in an unweighted graph. The algorithm
terminates because each of the `n` nodes is enqueued at most once.

## 4. Complexity

| Quantity | Cost |
| -------- | ---- |
| Time | `O(V + E)` |
| Extra space | `O(V)` for visited flags and the queue |

## 5. Implementation notes

- Neighbor order in `adj[u]` determines tie-breaking within a level.
- Works unchanged on undirected graphs if each undirected edge `{u,v}` appears
  as both `u→v` and `v→u`.
