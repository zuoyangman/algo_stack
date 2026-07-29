# KMP (Knuth–Morris–Pratt)

> Find occurrences of a pattern in a text in linear time using the longest proper prefix-suffix (LPS / π) table.

## Run

From `program_stack/`:

```bash
./scripts/run.sh strings/kmp java
./scripts/run.sh strings/kmp cpp
./scripts/run.sh strings/kmp rust
./scripts/run.sh strings/kmp go
```

## Public API (aligned across languages)

| Operation | Description |
| --------- | ----------- |
| `buildLps(pattern)` | LPS / failure function for `pattern` |
| `kmpSearch(text, pattern)` | All starting indices of matches |
| `kmpFirst(text, pattern)` | First match index, or `-1` |

## Notes

- Each language demo self-checks a few fixed cases and prints `kmp: ok`.

## Where to go next

- Theory → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending → [`EXTENSION.md`](EXTENSION.md)
