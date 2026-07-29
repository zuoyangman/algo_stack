# Topological Sort

> Produce a linear order of nodes in a DAG such that every edge goes forward; detect cycles via Kahn’s algorithm.

## Run

From `program_stack/`:

```bash
./scripts/run.sh graphs/topological_sort java
./scripts/run.sh graphs/topological_sort cpp
./scripts/run.sh graphs/topological_sort rust
./scripts/run.sh graphs/topological_sort go
```

## Public API (aligned across languages)

| Operation | Description |
| --------- | ----------- |
| `topologicalSort(adj)` | Kahn’s algorithm; returns order, or empty / null if a cycle exists |

## Notes

- Each language demo self-checks a few fixed cases and prints `topological_sort: ok`.

## Where to go next

- Theory → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending → [`EXTENSION.md`](EXTENSION.md)
