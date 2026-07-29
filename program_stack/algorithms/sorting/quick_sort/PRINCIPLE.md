# Quick Sort — Principle

## 1. Problem statement

Given an array of comparable keys, rearrange it into non-decreasing order.

## 2. Idea & invariants

Choose a pivot (here: last element). Partition so keys ≤ pivot lie to its
left and keys > pivot to its right. Recurse on each side.

**Invariant (Lomuto partition):** after scanning index `j`, indices
`lo..i` hold values ≤ pivot and `i+1..j` hold values > pivot.

## 3. Correctness sketch

Partition places the pivot in its final sorted position and separates
smaller/larger keys. By induction both sides become sorted, so the whole
array is sorted. Termination follows because each recursive call excludes
the pivot.

## 4. Complexity

| Quantity | Cost |
| -------- | ---- |
| Time (worst / average / best) | `O(n²)` / `O(n log n)` / `O(n log n)` |
| Extra space | `O(log n)` average stack; `O(n)` worst |

## 5. Implementation notes

In-place (aside from recursion stack). **Not stable** with the usual
swap-based partition. Worst case on already-sorted input with a naive
last-element pivot; random or median-of-three pivots mitigate this.
