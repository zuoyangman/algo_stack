# Queue — Extension Guide

## 1. Extension surface

Core ops: `enqueue`, `dequeue`, `front`, `is_empty`. Optional: `size`.

## 2. Common variants

### 2.1 Circular array queue
Fixed or growable ring buffer — better cache locality.

### 2.2 Deque
Allow insert/remove at both ends.

### 2.3 Priority queue
Order by priority (usually heap-backed), not arrival time.

## 3. Invariants to preserve

- `main` exits 0 only on success.
- Success line: `queue: ok`
- Cross-language API naming stays mappable.

## 4. Pitfalls

- Dequeue/front on empty queue.
- Forgetting to clear `tail` when dequeuing the last element.
- Ring-buffer off-by-one between full and empty states.

## 5. Suggested follow-up algorithms

- BFS
- Stack
- Sliding-window maximum (deque)
