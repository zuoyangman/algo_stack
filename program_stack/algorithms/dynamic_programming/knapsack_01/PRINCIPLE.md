# 0/1 Knapsack — Principle

## 1. Problem statement

Given `n` items with weights `w[i]` and values `v[i]`, and capacity `W`, choose
a subset of items with total weight `≤ W` maximizing total value. Each item may
be taken **0 or 1** time.

## 2. Idea & invariants

Define `dp[j]` = best value using capacity exactly at most `j` (1-D rolling).

Transition when considering item `i`:

```
for j = W downto w[i]:
    dp[j] = max(dp[j], dp[j - w[i]] + v[i])
```

Descending `j` ensures each item is used at most once (uses “old” `dp[j-w]`).

## 3. Correctness sketch

By induction on items considered: after processing the first `k` items, `dp[j]`
is optimal among subsets of those items with weight `≤ j`. Base: all zeros.

## 4. Complexity

| Quantity | Cost |
| -------- | ---- |
| Time | `O(nW)` |
| Extra space | `O(W)` with rolling array (`O(nW)` for full table) |

## 5. Implementation notes

- Weights / capacity should be non-negative integers.
- Reconstructing the chosen set needs the 2-D table or parent pointers.
