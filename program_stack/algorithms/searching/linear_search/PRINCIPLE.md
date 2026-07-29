# Linear Search — Principle

## 1. Problem statement

Given an array `a` and a target `x`, return the smallest index `i` with `a[i] == x`, or `-1` if no such index exists. The array need not be sorted.

## 2. Idea & invariants

Walk indices `0 .. n-1`. If `a[i]` equals `x`, return `i`. If the loop
ends without a hit, return `-1`.

**Invariant:** after checking index `i`, no earlier index holds `x`.

## 3. Correctness sketch

If a match exists, the first time the equality test succeeds is at the
leftmost such index. If none exists, every index was examined and rejected,
so `-1` is correct. The loop clearly terminates after at most `n` steps.

## 4. Complexity

| Quantity | Cost |
| -------- | ---- |
| Time (worst / average / best) | `O(n)` / `O(n)` / `O(1)` |
| Extra space | `O(1)` |

## 5. Implementation notes

Works on unsorted data. Prefer binary search when the array is sorted and
lookups dominate.
