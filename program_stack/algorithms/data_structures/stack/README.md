# Stack

> LIFO stack with push, pop, peek, and is_empty — demoed via parentheses balance and reverse.

## Run

From `program_stack/`:

```bash
./scripts/run.sh data_structures/stack java
./scripts/run.sh data_structures/stack cpp
./scripts/run.sh data_structures/stack rust
./scripts/run.sh data_structures/stack go
```

## Public API (aligned across languages)

| Operation | Description |
| --------- | ----------- |
| `push(x)` | Push onto the top |
| `pop()` | Remove and return the top |
| `peek()` / `top()` | Read the top without removing |
| `is_empty()` | Whether the stack has no elements |

Demo helpers: `is_balanced(s)` for `()[]{}`, and `reverse(s)`.

## Notes

- Each language demo self-checks a few fixed cases and prints `stack: ok`.

## Where to go next

- Theory → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending → [`EXTENSION.md`](EXTENSION.md)
