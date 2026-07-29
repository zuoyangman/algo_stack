# Quick Sort

> Partition around a pivot, then recursively sort the two sides.

## Run

From `program_stack/`:

```bash
./scripts/run.sh sorting/quick_sort java
./scripts/run.sh sorting/quick_sort cpp
./scripts/run.sh sorting/quick_sort rust
./scripts/run.sh sorting/quick_sort go
```

## Public API (aligned across languages)

| Operation | Description |
| --------- | ----------- |
| `sort(a)` | Sort integer array in non-decreasing order (in-place). |

## Notes

- Each language demo self-checks a few fixed cases and prints `quick_sort: ok`.

## Where to go next

- Theory → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending → [`EXTENSION.md`](EXTENSION.md)
