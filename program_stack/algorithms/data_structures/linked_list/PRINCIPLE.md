# Singly Linked List — Principle

## 1. Problem statement

Maintain an ordered sequence of values supporting insert/remove at the front,
append at the back, membership test, and enumeration — without contiguous
storage.

## 2. Idea & invariants

Each node holds a value and a pointer/reference to the next node. The list
keeps a `head` (and often a `tail` for O(1) append).

Invariants:

- The sequence of nodes reachable from `head` via `next` is the list contents.
- If `tail` is maintained: `tail.next == null` (or nil) when the list is non-empty;
  both `head` and `tail` are null when empty.
- `size` equals the number of reachable nodes.

## 3. Correctness sketch

- `push_front` links the new node before the old head; head updates atomically.
- `push_back` with a tail pointer attaches after `tail` and advances `tail`.
- `pop_front` advances `head`; if the list becomes empty, clear `tail`.
- `find` walks until a match or the end — terminates because the chain is finite
  and acyclic in a correct list.

## 4. Complexity

| Quantity | Cost |
| -------- | ---- |
| `push_front` / `pop_front` | `O(1)` |
| `push_back` (with tail) | `O(1)` |
| `find` / `to_array` | `O(n)` |
| Extra space | `O(n)` for `n` elements |

## 5. Implementation notes

Nodes are heap-allocated (or owned boxes in Rust). No random access; indexing
would be `O(n)`. Doubly linked lists add a `prev` pointer for O(1) back removal.
