# Hash Table — Principle

## 1. Problem statement

Associate keys with values under expected constant-time insert, lookup, and
delete, using a hash function into an array of buckets.

## 2. Idea & invariants

Separate chaining: each bucket is a list of `(key, value)` entries that hashed
to that index.

`index = hash(key) mod bucket_count` (non-negative).

On collision, walk the chain and compare keys for equality.

## 3. Correctness sketch

- `put` finds an equal key in the target chain and updates, or prepends a new
  entry.
- `get` / `remove` scan the same chain; only equal keys match.
- If the hash is fixed for a key and equality is an equivalence relation,
  lookups find the unique stored entry for that key (if any).

## 4. Complexity

| Quantity | Cost |
| -------- | ---- |
| Expected time (good hash, load factor α) | `O(1 + α)` |
| Worst time (all keys collide) | `O(n)` |
| Extra space | `O(n + B)` for `n` entries and `B` buckets |

## 5. Implementation notes

Resize (rehash) when load factor grows too high. Use a decent string hash
(e.g. polynomial rolling). Open addressing is an alternative to chaining.
