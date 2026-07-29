# 0/1 Knapsack

> Maximize total value of items packed into a capacity-limited knapsack, taking each item at most once.

## Run

From `program_stack/`:

```bash
./scripts/run.sh dynamic_programming/knapsack_01 java
./scripts/run.sh dynamic_programming/knapsack_01 cpp
./scripts/run.sh dynamic_programming/knapsack_01 rust
./scripts/run.sh dynamic_programming/knapsack_01 go
```

## Public API (aligned across languages)

| Operation | Description |
| --------- | ----------- |
| `knapsack01(weights, values, capacity)` | Returns maximum achievable value |

## Notes

- Each language demo self-checks a few fixed cases and prints `knapsack_01: ok`.

## Where to go next

- Theory → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending → [`EXTENSION.md`](EXTENSION.md)
