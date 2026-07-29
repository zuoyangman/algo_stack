# Bubble Sort — Extension Guide

## 1. Extension surface

The core `sort(a)` (and helpers used only by it) is safe to change. Keep `main`
as a pure self-check that prints `bubble_sort: ok` on success.

## 2. Common variants

### 2.1 Cocktail (bidirectional) bubble
Alternate left→right and right→left passes.

### 2.2 Odd–even sort
Parallel-friendly variant that compares odd/even index pairs.

## 3. Invariants to preserve

- `main` exits 0 only on success.
- Success line: `bubble_sort: ok`
- Cross-language API naming stays mappable.

## 4. Pitfalls

- Forgetting early-exit still works but loses the `O(n)` best case.
- Off-by-one on the shrinking upper bound can leave the last pair unchecked.

## 5. Suggested follow-up algorithms

- `insertion_sort`, `selection_sort` (other simple quadratic sorts)
- `quick_sort`, `merge_sort` (faster general-purpose sorts)
