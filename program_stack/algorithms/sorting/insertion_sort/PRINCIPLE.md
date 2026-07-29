# Insertion Sort — Principle

## 1. Problem statement

Given an array of comparable keys, rearrange it into non-decreasing order.

## 2. Idea & invariants

Maintain a sorted prefix `a[0..i)`. Take `key = a[i]` and shift larger
elements one step right until the hole for `key` is found, then write it.

**Invariant:** after step `i`, `a[0..i]` is a sorted permutation of the
original first `i+1` elements.

## 3. Correctness sketch

The invariant holds for `i = 0`. Shifting preserves relative order of the
already-sorted prefix and inserts `key` at the unique position that keeps
the prefix sorted. When `i` reaches `n-1` the whole array is sorted.

## 4. Complexity

| Quantity | Cost |
| -------- | ---- |
| Time (worst / average / best) | `O(n²)` / `O(n²)` / `O(n)` |
| Extra space | `O(1)` |

## 5. Implementation notes

In-place and **stable**. Excellent for nearly-sorted input and as the
base case inside hybrid sorts (e.g. Timsort / introsort).
