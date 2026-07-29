# Binary Search Tree

> Binary search tree with insert, contains, and inorder traversal (sorted order).

## Run

From `program_stack/`:

```bash
./scripts/run.sh data_structures/binary_search_tree java
./scripts/run.sh data_structures/binary_search_tree cpp
./scripts/run.sh data_structures/binary_search_tree rust
./scripts/run.sh data_structures/binary_search_tree go
```

## Public API (aligned across languages)

| Operation | Description |
| --------- | ----------- |
| `insert(x)` | Insert `x` (duplicates ignored or left-leaning — see notes) |
| `contains(x)` | Membership test |
| `inorder()` | Return values in sorted ascending order |

## Notes

- Each language demo self-checks a few fixed cases and prints `binary_search_tree: ok`.
- Duplicates are ignored on insert.

## Where to go next

- Theory → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending → [`EXTENSION.md`](EXTENSION.md)
