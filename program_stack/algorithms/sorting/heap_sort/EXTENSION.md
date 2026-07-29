# Heap Sort — Extension Guide

## 1. Extension surface

The core `sort(a)` (and helpers used only by it) is safe to change. Keep `main`
as a pure self-check that prints `heap_sort: ok` on success.

## 2. Common variants

### 2.1 Bottom-up heapsort
Use a bottom-up sift that walks to a leaf then sifts up.

### 2.2 Soft heaps / tournament trees
Related selection structures for other problems.

## 3. Invariants to preserve

- `main` exits 0 only on success.
- Success line: `heap_sort: ok`
- Cross-language API naming stays mappable.

## 4. Pitfalls

- Index arithmetic: left child `2i+1`, right `2i+2` (0-based).
- Build-heap must start at the last parent `(n/2)-1`, not at 0.
- Confusing min-heap with max-heap reverses the sort order.

## 5. Suggested follow-up algorithms

- `selection_sort` (same idea, linear select)
- `priority_queue` applications, `quick_sort`
