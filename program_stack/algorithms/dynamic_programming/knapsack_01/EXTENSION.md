# 0/1 Knapsack — Extension Guide

## 1. Extension surface

Core: `knapsack01(weights, values, capacity) → max value`. Safe to add: item
reconstruction, unbounded / bounded knapsack variants, fractional greedy note.

## 2. Common variants

### 2.1 Unbounded knapsack
Loop `j` ascending so an item can be reused.

### 2.2 Exact capacity / count items
Track secondary objectives or force `dp` on exact weight.

### 2.3 Meet-in-the-middle
For small `n` (~40) and large `W`, split items into two halves.

## 3. Invariants to preserve

- `main` exits 0 only on success.
- Success line: `knapsack_01: ok`
- Cross-language API naming stays mappable.

## 4. Pitfalls

- Ascending inner loop accidentally implements unbounded knapsack.
- Integer overflow on value sums for large inputs.
- `W = 0` or empty item list should return 0.

## 5. Suggested follow-up algorithms

- Unbounded knapsack, subset sum, coin change, partition equal subset sum
