# Dijkstra’s Algorithm

> Compute shortest-path distances from a source on a directed graph with non-negative edge weights.

## Run

From `program_stack/`:

```bash
./scripts/run.sh graphs/dijkstra java
./scripts/run.sh graphs/dijkstra cpp
./scripts/run.sh graphs/dijkstra rust
./scripts/run.sh graphs/dijkstra go
```

## Public API (aligned across languages)

| Operation | Description |
| --------- | ----------- |
| `dijkstra(n, edges, source)` | Returns distance array; unreachable = a large sentinel (`INF`) |

`edges` is a list of `(u, v, w)` triples. This demo uses the simple **O(V²)**
“scan for minimum” implementation (no heap required).

## Notes

- Each language demo self-checks a few fixed cases and prints `dijkstra: ok`.

## Where to go next

- Theory → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending → [`EXTENSION.md`](EXTENSION.md)
