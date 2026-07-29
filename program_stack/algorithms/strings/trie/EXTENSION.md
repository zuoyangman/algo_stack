# Trie — Extension Guide

## 1. Extension surface

Core type `Trie` with `insert` / `search` / `startsWith`. Safe to add: delete,
count of words with prefix, autocomplete listing, compressed radix tree.

## 2. Common variants

### 2.1 Fixed alphabet array
`children[26]` for `'a'..'z'` — less overhead than hash maps.

### 2.2 Radix / Patricia tree
Compress unary paths for memory and cache locality.

### 2.3 XOR trie / binary trie
Bits as edges — useful for maximum XOR pair queries.

## 3. Invariants to preserve

- `main` exits 0 only on success.
- Success line: `trie: ok`
- Cross-language API naming stays mappable.

## 4. Pitfalls

- `search("app")` false after only inserting `"apple"` (prefix ≠ word).
- Not distinguishing end-of-word from intermediate nodes.
- Empty string: decide whether root is an end node.

## 5. Suggested follow-up algorithms

- Aho–Corasick, suffix trie / suffix tree (toy), autocomplete systems
