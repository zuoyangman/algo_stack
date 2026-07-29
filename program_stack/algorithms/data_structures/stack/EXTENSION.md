# Stack — Extension Guide

## 1. Extension surface

Core ops: `push`, `pop`, `peek`, `is_empty`. Demos may add `is_balanced` /
`reverse` without changing the core contract.

## 2. Common variants

### 2.1 Min-stack
Track current minimum in O(1) with an auxiliary stack of mins.

### 2.2 Linked stack
Use nodes instead of a growable array — no amortized resize.

### 2.3 Bounded stack
Fixed capacity; push fails or grows by policy when full.

## 3. Invariants to preserve

- `main` exits 0 only on success.
- Success line: `stack: ok`
- Cross-language API naming stays mappable.

## 4. Pitfalls

- Popping an empty stack (undefined / panic / exception).
- Matching brackets: only compare against the top, not an arbitrary opener.
- Integer overflow if storing large counts separately from the backing store.

## 5. Suggested follow-up algorithms

- Queue / Deque
- Expression evaluation (shunting yard)
- DFS with an explicit stack
