# Hash Table (Separate Chaining)

> Hash map with separate chaining: put, get, and remove for string keys.

## Run

From `program_stack/`:

```bash
./scripts/run.sh data_structures/hash_table java
./scripts/run.sh data_structures/hash_table cpp
./scripts/run.sh data_structures/hash_table rust
./scripts/run.sh data_structures/hash_table go
```

## Public API (aligned across languages)

| Operation | Description |
| --------- | ----------- |
| `put(key, value)` | Insert or update |
| `get(key)` | Lookup; indicate missing |
| `remove(key)` | Delete if present |

## Notes

- Each language demo self-checks a few fixed cases and prints `hash_table: ok`.
- Buckets are linked lists (separate chaining). Keys are strings; values are ints.

## Where to go next

- Theory → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending → [`EXTENSION.md`](EXTENSION.md)
