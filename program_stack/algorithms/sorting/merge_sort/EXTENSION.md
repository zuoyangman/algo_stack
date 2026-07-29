# Merge Sort — Extension Guide

## 1. Extension surface

The core `sort(a)` (and helpers used only by it) is safe to change. Keep `main`
as a pure self-check that prints `merge_sort: ok` on success.

## 2. Common variants

### 2.1 Bottom-up merge sort
Iteratively merge runs of size 1, 2, 4, …

### 2.2 Natural merge sort
Exploit already-sorted runs in the input.

## 3. Invariants to preserve

- `main` exits 0 only on success.
- Success line: `merge_sort: ok`
- Cross-language API naming stays mappable.

## 4. Pitfalls

- Midpoint overflow: prefer `lo + (hi - lo) / 2` over `(lo + hi) / 2`.
- Stability requires taking from the left run when keys compare equal.
- Forgetting to copy the leftover tail of a run corrupts the result.

## 5. Suggested follow-up algorithms

- `quick_sort`, `heap_sort`
- `tim_sort` (natural + galloping merges)
