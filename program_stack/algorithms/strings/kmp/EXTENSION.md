# KMP — Extension Guide

## 1. Extension surface

Core: `buildLps`, `kmpSearch`, `kmpFirst`. Safe to add: streaming search,
count-only API, case-insensitive wrapper.

## 2. Common variants

### 2.1 String matching automaton
Explicit transition table on alphabet Σ (faster per char, more memory).

### 2.2 Z-algorithm
Related linear-time matching via Z-boxes.

### 2.3 Multiple patterns
Aho–Corasick generalizes KMP’s failure links to a trie.

## 3. Invariants to preserve

- `main` exits 0 only on success.
- Success line: `kmp: ok`
- Cross-language API naming stays mappable.

## 4. Pitfalls

- Off-by-one when writing LPS (`len` vs `lps[len-1]`).
- Forgetting to continue after a match (`q = lps[q-1]`).
- Naive `O(nm)` fallback accidentally used in tests for large inputs.

## 5. Suggested follow-up algorithms

- Rabin–Karp, Boyer–Moore, Aho–Corasick, Z-algorithm
