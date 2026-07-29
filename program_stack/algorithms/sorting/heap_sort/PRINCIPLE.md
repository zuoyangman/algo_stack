# Heap Sort — Principle

## 1. Problem statement

Given an array of comparable keys, rearrange it into non-decreasing order.

## 2. Idea & invariants

Interpret the array as a binary heap. Sift down from the last parent to
build a max-heap. Then swap the root (maximum) with the end of the heap,
shrink the heap, and sift the new root down — repeating until one element
remains.

**Invariant:** after `k` extractions, `a[n-k..n)` holds the `k` largest
keys in sorted order, and `a[0..n-k)` is a max-heap.

## 3. Correctness sketch

Heapify establishes the max-heap property in linear time. Each extract-max
places the next-largest key into its final position and restores the heap
on a smaller prefix, so the suffix grows correctly sorted.

## 4. Complexity

| Quantity | Cost |
| -------- | ---- |
| Time (worst / average / best) | `O(n log n)` / `O(n log n)` / `O(n log n)` |
| Extra space | `O(1)` |

## 5. Implementation notes

In-place and **not stable**. Guaranteed `O(n log n)` without the
quicksort worst case; typically slower than well-tuned quicksort due to
poor cache locality.
