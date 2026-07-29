# Insertion Sort — Extension Guide

## 1. Extension surface

The core `sort(a)` (and helpers used only by it) is safe to change. Keep `main`
as a pure self-check that prints `insertion_sort: ok` on success.

## 2. Common variants

### 2.1 Binary insertion
Use binary search to find the insertion point (still `O(n²)` moves).

### 2.2 Shell sort
Insertion sort on gapped subsequences.

## 3. Invariants to preserve

- `main` exits 0 only on success.
- Success line: `insertion_sort: ok`
- Cross-language API naming stays mappable.

## 4. Pitfalls

- Shifting must run right-to-left; left-to-right overwrites unread values.
- Stability requires `>` (not `>=`) when deciding to shift.

## 5. Suggested follow-up algorithms

- `bubble_sort`, `selection_sort`
- `shell_sort`, `merge_sort`
