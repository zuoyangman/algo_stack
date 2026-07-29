# Selection Sort

> Repeatedly select the minimum of the unsorted suffix and swap it into place.

## Run

From `program_stack/`:

```bash
./scripts/run.sh sorting/selection_sort java
./scripts/run.sh sorting/selection_sort cpp
./scripts/run.sh sorting/selection_sort rust
./scripts/run.sh sorting/selection_sort go
```

## Public API (aligned across languages)

| Operation | Description |
| --------- | ----------- |
| `sort(a)` | Sort integer array in non-decreasing order (in-place). |

## Notes

- Each language demo self-checks a few fixed cases and prints `selection_sort: ok`.

## Where to go next

- Theory → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending → [`EXTENSION.md`](EXTENSION.md)
