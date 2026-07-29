# Singly Linked List — Extension Guide

## 1. Extension surface

Safe to extend: node type, `push_front` / `push_back` / `pop_front` / `find` /
`values` (or `to_array`). Keep `main` self-checks and the success line.

## 2. Common variants

### 2.1 Doubly linked list
Add `prev` on each node; enable O(1) `pop_back` and bidirectional iteration.

### 2.2 Sentinel / dummy head
Use a dummy node so insert/delete never special-case an empty list.

### 2.3 Circular list
Last node points to head; useful for round-robin scheduling demos.

## 3. Invariants to preserve

- `main` exits 0 only on success.
- Success line: `linked_list: ok`
- Cross-language API naming stays mappable.

## 4. Pitfalls

- Forgetting to update `tail` when popping the last element.
- Creating cycles by incorrect `next` wiring.
- Use-after-free in C++ if nodes are deleted while still linked.
- Rust ownership: prefer `Option<Box<Node>>` for singly linked lists.

## 5. Suggested follow-up algorithms

- Stack / Queue (list-backed)
- Skip list
- LRU cache (doubly linked + hash map)
