# Longest Common Subsequence (LCS)

> Compute the length of the longest subsequence common to two strings (and optionally recover one such string).

## Run

From `program_stack/`:

```bash
./scripts/run.sh dynamic_programming/lcs java
./scripts/run.sh dynamic_programming/lcs cpp
./scripts/run.sh dynamic_programming/lcs rust
./scripts/run.sh dynamic_programming/lcs go
```

## Public API (aligned across languages)

| Operation | Description |
| --------- | ----------- |
| `lcsLength(a, b)` | Length of an LCS of strings `a` and `b` |
| `lcsString(a, b)` | One concrete LCS string (any valid) |

## Notes

- Each language demo self-checks a few fixed cases and prints `lcs: ok`.

## Where to go next

- Theory → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending → [`EXTENSION.md`](EXTENSION.md)
