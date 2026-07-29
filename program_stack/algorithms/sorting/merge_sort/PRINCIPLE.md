# Merge Sort — Principle

## 1. Problem statement

Given an array of comparable keys, rearrange it into non-decreasing order.

## 2. Idea & invariants

Split the array into two halves, sort each recursively, then merge the two
sorted runs into one sorted run by repeatedly taking the smaller head.

**Invariant (merge):** the output built so far is sorted and contains the
smallest `k` elements from the two input runs.

## 3. Correctness sketch

Base case of length ≤ 1 is sorted. By induction both halves are sorted;
the merge of two sorted sequences is sorted and contains exactly the same
multiset of keys. Recursion depth is finite because the range shrinks.

## 4. Complexity

| Quantity | Cost |
| -------- | ---- |
| Time (worst / average / best) | `O(n log n)` / `O(n log n)` / `O(n log n)` |
| Extra space | `O(n)` for the merge buffer |

## 5. Implementation notes

Out-of-place (needs a temporary buffer). **Stable** if the merge prefers
the left run on ties. Recursion depth `O(log n)` for balanced splits.
