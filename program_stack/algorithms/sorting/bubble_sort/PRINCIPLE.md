# Bubble Sort — Principle

## 1. Problem statement

Given an array of comparable keys, rearrange it into non-decreasing order.

## 2. Idea & invariants

Scan the array left-to-right, swapping every adjacent pair that is out of
order. After one full pass the largest element is in place (the “bubble”).
Repeat, shrinking the unsorted prefix, until a pass makes no swaps.

**Invariant:** after pass `k`, the last `k` elements are the `k` largest and
are in final sorted order.

## 3. Correctness sketch

Each swap fixes an inversion. There are finitely many inversions, so the
algorithm terminates. When a pass finds no inversion the array is sorted.
The final-position argument shows the largest remaining key moves to the
end of the unsorted region each pass.

## 4. Complexity

| Quantity | Cost |
| -------- | ---- |
| Time (worst / average / best) | `O(n²)` / `O(n²)` / `O(n)` with early exit |
| Extra space | `O(1)` |

## 5. Implementation notes

In-place and **stable** (equal keys are never swapped past each other).
Early-exit when a pass does no swaps gives best-case linear time.
