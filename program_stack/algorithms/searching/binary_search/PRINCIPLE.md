# Binary Search — Principle

## 1. Problem statement

Given a **sorted** (non-decreasing) array `a` and target `x`, return some index `i` with `a[i] == x`, or `-1` if `x` is not present.

## 2. Idea & invariants

Maintain a half-open or closed window `[lo, hi]` that must contain `x` if
it exists. Compare `a[mid]` to `x`: move `lo` right of `mid` if too small,
`hi` left of `mid` if too large, or return `mid` on equality.

**Invariant:** if `x` occurs in the original array, it occurs somewhere in
the current window.

## 3. Correctness sketch

The window always contains every remaining candidate. Each step strictly
shrinks the window, so the search terminates. When the window is empty,
no candidate remains and the answer is `-1`. Equality returns a valid index.

## 4. Complexity

| Quantity | Cost |
| -------- | ---- |
| Time (worst / average / best) | `O(log n)` / `O(log n)` / `O(1)` |
| Extra space | `O(1)` iterative |

## 5. Implementation notes

Requires sorted input (this demo does not verify that). Iterative form
avoids recursion depth issues. With duplicates, any matching index is valid;
lower/upper-bound variants pick the leftmost or rightmost.
