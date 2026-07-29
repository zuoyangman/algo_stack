# Trie (Prefix Tree)

> Insert, exact search, and prefix (`startsWith`) queries over a dynamic set of strings.

## Run

From `program_stack/`:

```bash
./scripts/run.sh strings/trie java
./scripts/run.sh strings/trie cpp
./scripts/run.sh strings/trie rust
./scripts/run.sh strings/trie go
```

## Public API (aligned across languages)

| Operation | Description |
| --------- | ----------- |
| `insert(word)` | Add `word` to the trie |
| `search(word)` | True iff `word` was inserted |
| `startsWith(prefix)` | True iff some inserted word has this prefix |

## Notes

- Each language demo self-checks a few fixed cases and prints `trie: ok`.
- Nodes use a map/dictionary of children (alphabet-agnostic, dependency-free).

## Where to go next

- Theory → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending → [`EXTENSION.md`](EXTENSION.md)
