# Stack — Principle

## 1. Problem statement

Provide last-in, first-out (LIFO) access: insert and remove only at one end
(the *top*).

## 2. Idea & invariants

Backed by a dynamic array (or linked list). The top is the last inserted
element still present.

Invariant: after any sequence of push/pop, the sequence of remaining elements
from bottom to top equals the subsequence of unmatched pushes.

## 3. Correctness sketch

- `push` appends; `pop`/`peek` read the last index — LIFO by construction.
- Balanced parentheses: push openings; on a closing bracket, the top must be
  the matching opener. An empty stack at the end means every opener was matched
  in nested/sequential order.

## 4. Complexity

| Quantity | Cost |
| -------- | ---- |
| `push` / `pop` / `peek` / `is_empty` | Amortized `O(1)` (array) |
| Extra space | `O(n)` for `n` elements |

## 5. Implementation notes

Array-backed stacks resize; linked stacks allocate per push. Prefer array for
cache locality in demos. Always check emptiness before `pop`/`peek`.
