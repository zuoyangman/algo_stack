# DFS — Principle

## 1. Problem statement

Given an adjacency-list graph and a start node `s`, return the order in which
nodes are first visited when performing depth-first search from `s`.

## 2. Idea & invariants

- From the current node, recursively visit each unvisited neighbor in list order.
- Mark a node visited when first entered (preorder).
- Stack of recursive calls (or an explicit stack) holds the current path.

## 3. Correctness sketch

Every reachable node is eventually entered because when a node finishes, DFS
continues with the next unexplored neighbor of an ancestor. Each node is marked
visited once, so the recursion terminates.

## 4. Complexity

| Quantity | Cost |
| -------- | ---- |
| Time | `O(V + E)` |
| Extra space | `O(V)` for visited + recursion / stack depth |

## 5. Implementation notes

- This demo uses recursive DFS (preorder). Iterative DFS with an explicit stack
  can differ in visit order depending on push order.
- Deep graphs may overflow the call stack; switch to iterative DFS if needed.
