# Selection Sort — Principle

## 1. Problem statement

Given an array of comparable keys, rearrange it into non-decreasing order.

## 2. Idea & invariants

For `i` from `0` to `n-2`, find the index of the minimum in `a[i..n)` and
swap it with `a[i]`.

**Invariant:** after step `i`, `a[0..i)` holds the `i` smallest elements in
sorted order, and `a[i..n)` holds the rest.

## 3. Correctness sketch

Selecting the true minimum of the remaining suffix and placing it at
position `i` extends the sorted prefix by one correctly ordered element.
After `n-1` such steps the last element is automatically in place.

## 4. Complexity

| Quantity | Cost |
| -------- | ---- |
| Time (worst / average / best) | `O(n²)` / `O(n²)` / `O(n²)` |
| Extra space | `O(1)` |

## 5. Implementation notes

In-place but **not stable** (a swap can jump an equal key past another).
Always Θ(n²) comparisons; useful when writes are expensive (at most n−1 swaps).
