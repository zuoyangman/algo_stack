# Binary Search

> Find a target in a sorted array by repeatedly discarding half the range.

## Run

From `program_stack/`:

```bash
./scripts/run.sh searching/binary_search java
./scripts/run.sh searching/binary_search cpp
./scripts/run.sh searching/binary_search rust
./scripts/run.sh searching/binary_search go
```

## Public API (aligned across languages)

| Operation | Description |
| --------- | ----------- |
| `search(a, x)` | On a sorted array, return an index of `x`, or `-1` if absent. |

## Notes

- Each language demo self-checks a few fixed cases and prints `binary_search: ok`.

## Where to go next

- Theory → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending → [`EXTENSION.md`](EXTENSION.md)
