# Union–Find (Disjoint Set Union)

> DSU with path compression and union by rank: find, union, and connected.

## Run

From `program_stack/`:

```bash
./scripts/run.sh data_structures/union_find java
./scripts/run.sh data_structures/union_find cpp
./scripts/run.sh data_structures/union_find rust
./scripts/run.sh data_structures/union_find go
```

## Public API (aligned across languages)

| Operation | Description |
| --------- | ----------- |
| `find(x)` | Representative (root) of `x`'s set |
| `union(a, b)` | Merge sets containing `a` and `b` |
| `connected(a, b)` | Whether `a` and `b` share a set |

Elements are integers in `0 .. n-1`.

## Notes

- Each language demo self-checks a few fixed cases and prints `union_find: ok`.

## Where to go next

- Theory → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending → [`EXTENSION.md`](EXTENSION.md)
