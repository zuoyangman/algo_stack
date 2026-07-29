# KMP — Principle

## 1. Problem statement

Given text `T` of length `n` and pattern `P` of length `m`, find all starting
indices `i` where `T[i..i+m) = P`. Empty pattern is treated as matching at
every position or handled as a special case (demos use non-empty patterns and
return `[-1]` semantics for first-match on empty via explicit checks).

## 2. Idea & invariants

Precompute `lps[i]` = longest proper prefix of `P[0..i]` that is also a suffix.
While scanning `T`, on a mismatch jump the pattern index using `lps` instead of
restarting from scratch — never re-examine earlier text characters.

## 3. Correctness sketch

After a partial match of length `q` fails at `P[q]`, any viable realignment
must be a border of `P[0..q)`. `lps[q-1]` is the longest such border, so
skipping to it loses no match. Building LPS is the same automaton idea on `P`
itself.

## 4. Complexity

| Quantity | Cost |
| -------- | ---- |
| Time | `O(n + m)` |
| Extra space | `O(m)` for LPS |

## 5. Implementation notes

- Careful with empty pattern / empty text edge cases.
- Overlapping matches (e.g. `"aaa"` in `"aaaa"`) are found by setting
  `q = lps[q-1]` after a full match.
