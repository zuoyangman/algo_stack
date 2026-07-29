# Singly Linked List

> A classic singly linked list with push/pop at the front, push at the back, find, and values export.

## Run

From `program_stack/`:

```bash
./scripts/run.sh data_structures/linked_list java
./scripts/run.sh data_structures/linked_list cpp
./scripts/run.sh data_structures/linked_list rust
./scripts/run.sh data_structures/linked_list go
```

## Public API (aligned across languages)

| Operation | Description |
| --------- | ----------- |
| `push_front(x)` | Insert `x` at the head |
| `push_back(x)` | Insert `x` at the tail |
| `pop_front()` | Remove and return the head value |
| `find(x)` | Return whether `x` is present |
| `to_array` / `values()` | Collect elements head→tail |

## Notes

- Each language demo self-checks a few fixed cases and prints `linked_list: ok`.

## Where to go next

- Theory → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending → [`EXTENSION.md`](EXTENSION.md)
