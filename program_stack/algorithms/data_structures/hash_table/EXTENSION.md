# Hash Table — Extension Guide

## 1. Extension surface

Core: `put`, `get`, `remove`, plus the hash function and bucket array.
Optional: `contains`, `size`, resize policy.

## 2. Common variants

### 2.1 Open addressing
Linear/quadratic probing or double hashing — no per-bucket lists.

### 2.2 Resize / rehash
Double bucket count when `n/B` exceeds a threshold (e.g. 0.75).

### 2.3 Integer keys
Replace string hash with identity or mix functions (murmur-style mix).

## 3. Invariants to preserve

- `main` exits 0 only on success.
- Success line: `hash_table: ok`
- Cross-language API naming stays mappable.

## 4. Pitfalls

- Negative hash values when taking modulo in languages with signed hashes.
- Mutating keys that affect equality/hash after insertion.
- Forgetting to update vs. insert on duplicate keys.

## 5. Suggested follow-up algorithms

- Bloom filter
- LRU cache (hash + doubly linked list)
- Consistent hashing
