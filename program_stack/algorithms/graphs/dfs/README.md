# DFS (Depth-First Search)

> Explore a graph by going as deep as possible along each branch before backtracking; return the visit order.

## Run

From `program_stack/`:

```bash
./scripts/run.sh graphs/dfs java
./scripts/run.sh graphs/dfs cpp
./scripts/run.sh graphs/dfs rust
./scripts/run.sh graphs/dfs go
```

## Public API (aligned across languages)

| Operation | Description |
| --------- | ----------- |
| `dfs(adj, start)` | Recursive DFS; returns preorder visit order of reachable nodes |

## Notes

- Each language demo self-checks a few fixed cases and prints `dfs: ok`.

## Where to go next

- Theory → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending → [`EXTENSION.md`](EXTENSION.md)
