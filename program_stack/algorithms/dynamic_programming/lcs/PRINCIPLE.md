# LCS — Principle

## 1. Problem statement

A **subsequence** is obtained by deleting zero or more characters without
reordering. Given strings `A[1..n]` and `B[1..m]`, find the length of a longest
common subsequence (and optionally one such string).

## 2. Idea & invariants

```
dp[i][j] = LCS length of A[1..i] and B[1..j]

dp[i][j] = dp[i-1][j-1] + 1           if A[i] == B[j]
         = max(dp[i-1][j], dp[i][j-1]) otherwise
```

Base: `dp[0][*] = dp[*][0] = 0`.

## 3. Correctness sketch

If last characters match they belong to some LCS of the prefixes; otherwise an
LCS drops the last char of `A` or of `B`. Optimal substructure + overlapping
subproblems yield the recurrence.

## 4. Complexity

| Quantity | Cost |
| -------- | ---- |
| Time | `O(nm)` |
| Extra space | `O(nm)` full table; `O(min(n,m))` for length-only rolling |

## 5. Implementation notes

- Reconstruction walks the DP table diagonally on matches.
- Multiple LCS strings may exist; demos accept any of maximum length.
