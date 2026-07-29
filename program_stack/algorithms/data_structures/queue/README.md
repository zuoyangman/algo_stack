# Queue

> FIFO queue with enqueue, dequeue, front, and is_empty.

## Run

From `program_stack/`:

```bash
./scripts/run.sh data_structures/queue java
./scripts/run.sh data_structures/queue cpp
./scripts/run.sh data_structures/queue rust
./scripts/run.sh data_structures/queue go
```

## Public API (aligned across languages)

| Operation | Description |
| --------- | ----------- |
| `enqueue(x)` | Insert at the back |
| `dequeue()` | Remove and return the front |
| `front()` / `peek()` | Read the front without removing |
| `is_empty()` | Whether the queue has no elements |

## Notes

- Each language demo self-checks a few fixed cases and prints `queue: ok`.

## Where to go next

- Theory → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending → [`EXTENSION.md`](EXTENSION.md)
