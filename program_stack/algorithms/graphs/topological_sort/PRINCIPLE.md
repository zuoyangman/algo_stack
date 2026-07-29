# Topological Sort — Principle

## 1. Problem statement

Given a directed graph, return a permutation of vertices such that for every
edge `u → v`, `u` appears before `v`. If the graph contains a cycle, report
failure (empty result).

## 2. Idea & invariants (Kahn)

- Compute indegree of every node.
- Enqueue all nodes with indegree 0.
- Repeatedly dequeue `u`, append to the order, and decrement indegrees of
  neighbors; enqueue those that reach 0.
- Invariant: every node in the queue has all predecessors already ordered.

## 3. Correctness sketch

On a DAG every node eventually reaches indegree 0, so `|order| = V`. If a
cycle exists, at least one node never reaches indegree 0, so `|order| < V`.

## 4. Complexity

| Quantity | Cost |
| -------- | ---- |
| Time | `O(V + E)` |
| Extra space | `O(V)` |

## 5. Implementation notes

- Queue FIFO vs. priority queue changes which valid topo order is produced.
- DFS-based topo sort (finish-time reverse) is an equally valid alternative.
