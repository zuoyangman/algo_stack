# Queue — Principle

## 1. Problem statement

Provide first-in, first-out (FIFO) access: insert at the back, remove from the
front.

## 2. Idea & invariants

A linked list with head (front) and tail (back) pointers, or a circular buffer
on an array. Elements leave in the same order they arrived.

Invariant: the sequence from front to back equals the order of unmatched
enqueues.

## 3. Correctness sketch

- `enqueue` attaches after `tail` and advances `tail`.
- `dequeue` / `front` operate on `head`.
- Empty when `head` is null (linked) or head index equals tail (ring buffer).

## 4. Complexity

| Quantity | Cost |
| -------- | ---- |
| `enqueue` / `dequeue` / `front` / `is_empty` | `O(1)` |
| Extra space | `O(n)` |

## 5. Implementation notes

Array-backed queues need a circular buffer (or occasional compact) to avoid
O(n) shifts. Linked queues are simpler for didactic demos.
