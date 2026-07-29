# Selection Sort — Extension Guide

## 1. Extension surface

The core `sort(a)` (and helpers used only by it) is safe to change. Keep `main`
as a pure self-check that prints `selection_sort: ok` on success.

## 2. Common variants

### 2.1 Double selection
Track min and max each pass to halve the number of passes.

### 2.2 Heap sort
Same “select extremum” idea with a heap for `O(n log n)`.

## 3. Invariants to preserve

- `main` exits 0 only on success.
- Success line: `selection_sort: ok`
- Cross-language API naming stays mappable.

## 4. Pitfalls

- Stability is lost unless you shift instead of swap.
- Best case is still quadratic — no early exit helps comparisons.

## 5. Suggested follow-up algorithms

- `bubble_sort`, `insertion_sort`
- `heap_sort`
