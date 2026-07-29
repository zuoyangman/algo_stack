# LCS — Extension Guide

## 1. Extension surface

Core: `lcsLength` / `lcsString`. Safe to add: all LCS enumeration, LCS of more
than two strings, Hirschberg’s linear-space reconstruction.

## 2. Common variants

### 2.1 Longest common substring
Contiguous — different DP (reset on mismatch).

### 2.2 Edit distance
Related DP table with insert/delete/substitute costs.

### 2.3 Diff / patience
LCS underpins classic text-diff algorithms.

## 3. Invariants to preserve

- `main` exits 0 only on success.
- Success line: `lcs: ok`
- Cross-language API naming stays mappable.

## 4. Pitfalls

- Confusing subsequence with substring.
- Off-by-one when indexing 0-based strings into 1-based DP.
- Reconstructing without checking both up and left branches.

## 5. Suggested follow-up algorithms

- Edit distance, LIS (via patience / DP), longest palindromic subsequence
