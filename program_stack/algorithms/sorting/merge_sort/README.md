# Merge Sort

> Divide-and-conquer sort: recursively sort halves, then merge them.

## Run

From `program_stack/`:

```bash
./scripts/run.sh sorting/merge_sort java
./scripts/run.sh sorting/merge_sort cpp
./scripts/run.sh sorting/merge_sort rust
./scripts/run.sh sorting/merge_sort go
```

## Public API (aligned across languages)

| Operation | Description |
| --------- | ----------- |
| `sort(a)` | Sort integer array in non-decreasing order. |

## Notes

- Each language demo self-checks a few fixed cases and prints `merge_sort: ok`.

## Where to go next

- Theory → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending → [`EXTENSION.md`](EXTENSION.md)
