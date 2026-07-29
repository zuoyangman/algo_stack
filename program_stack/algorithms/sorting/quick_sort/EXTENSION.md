# Quick Sort — Extension Guide

## 1. Extension surface

The core `sort(a)` (and helpers used only by it) is safe to change. Keep `main`
as a pure self-check that prints `quick_sort: ok` on success.

## 2. Common variants

### 2.1 Hoare partition
Two converging pointers; fewer swaps on average.

### 2.2 3-way (Dutch National Flag)
Handle many duplicates by splitting `< / = / >` pivot.

### 2.3 Introsort
Fall back to heap sort when recursion depth is excessive.

## 3. Invariants to preserve

- `main` exits 0 only on success.
- Success line: `quick_sort: ok`
- Cross-language API naming stays mappable.

## 4. Pitfalls

- Off-by-one when excluding the pivot from recursive ranges.
- Duplicate-heavy arrays degrade Lomuto; prefer 3-way partition.
- Stack overflow on adversarial inputs without depth limiting.

## 5. Suggested follow-up algorithms

- `merge_sort`, `heap_sort`
- `intro_sort`, `dual_pivot_quicksort`
