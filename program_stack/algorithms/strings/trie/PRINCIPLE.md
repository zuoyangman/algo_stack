# Trie — Principle

## 1. Problem statement

Maintain a dynamic set of strings supporting:

- `insert(word)`
- `search(word)` — exact membership
- `startsWith(prefix)` — whether any word begins with `prefix`

## 2. Idea & invariants

Each edge is labeled by a character; a path from the root spells a string.
Nodes marked `end` indicate that the path is a complete inserted word.
Invariant: every inserted word corresponds to a unique root-to-`end` path.

## 3. Correctness sketch

`insert` creates missing edges along the word and marks the final node.
`search` follows edges and requires the final node to be marked `end`.
`startsWith` only requires that the path exists (end mark optional).

## 4. Complexity

| Quantity | Cost |
| -------- | ---- |
| Time per op | `O(L)` for word/prefix length `L` |
| Extra space | `O(total characters stored)` worst case |

## 5. Implementation notes

- Map-based children support any character set; array-of-26 is faster for
  lowercase English only.
- Deletion is omitted in this didactic demo (can mark `end=false` and prune).
