# Binary Search — Extension Guide

## 1. Extension surface

The core `search(a, x)` (and helpers used only by it) is safe to change. Keep `main`
as a pure self-check that prints `binary_search: ok` on success.

## 2. Common variants

### 2.1 Lower / upper bound
First index `≥ x` or first index `> x` (C++ `lower_bound` / `upper_bound`).

### 2.2 Binary search on answer
Search a monotonic predicate over a numeric domain, not an array.

## 3. Invariants to preserve

- `main` exits 0 only on success.
- Success line: `binary_search: ok`
- Cross-language API naming stays mappable.

## 4. Pitfalls

- Midpoint overflow: use `lo + (hi - lo) / 2`.
- Off-by-one when updating `lo`/`hi` after comparing with `mid`.
- Calling on an unsorted array yields undefined/wrong results.

## 5. Suggested follow-up algorithms

- `linear_search`
- `binary_search_tree`, exponential search, interpolation search
