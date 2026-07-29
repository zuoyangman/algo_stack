# BFS (Breadth-First Search)

> Traverse a graph level by level from a start node and return the visit order.

## Run

From `program_stack/`:

```bash
./scripts/run.sh graphs/bfs java
./scripts/run.sh graphs/bfs cpp
./scripts/run.sh graphs/bfs rust
./scripts/run.sh graphs/bfs go
```

## Public API (aligned across languages)

| Operation | Description |
| --------- | ----------- |
| `bfs(adj, start)` | Adjacency-list BFS; returns visit order of reachable nodes |

Graph is `List[List[int]]` / `vector<vector<int>>` / `Vec<Vec<usize>>` / `[][]int` — node `u`’s neighbors are `adj[u]`.

## Notes

- Each language demo self-checks a few fixed cases and prints `bfs: ok`.

## Where to go next

- Theory → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending → [`EXTENSION.md`](EXTENSION.md)
