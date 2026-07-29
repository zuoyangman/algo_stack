# Heap Sort

> Build a max-heap, then repeatedly extract the maximum into the sorted suffix.

## Run

From `program_stack/`:

```bash
./scripts/run.sh sorting/heap_sort java
./scripts/run.sh sorting/heap_sort cpp
./scripts/run.sh sorting/heap_sort rust
./scripts/run.sh sorting/heap_sort go
```

## Public API (aligned across languages)

| Operation | Description |
| --------- | ----------- |
| `sort(a)` | Sort integer array in non-decreasing order (in-place). |

## Notes

- Each language demo self-checks a few fixed cases and prints `heap_sort: ok`.

## Where to go next

- Theory → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending → [`EXTENSION.md`](EXTENSION.md)
