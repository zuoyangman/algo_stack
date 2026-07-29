# Linear Search — Extension Guide

## 1. Extension surface

The core `search(a, x)` (and helpers used only by it) is safe to change. Keep `main`
as a pure self-check that prints `linear_search: ok` on success.

## 2. Common variants

### 2.1 Sentinel search
Place `x` at the end to avoid a bounds check each step.

### 2.2 Parallel / blocked scan
SIMD or cache-blocked linear scans for large arrays.

## 3. Invariants to preserve

- `main` exits 0 only on success.
- Success line: `linear_search: ok`
- Cross-language API naming stays mappable.

## 4. Pitfalls

- Returning any match vs. the first match — demos require the first.
- Forgetting the empty-array case (should return `-1`).

## 5. Suggested follow-up algorithms

- `binary_search`
- hashing / `hash_table` for average `O(1)` lookups
