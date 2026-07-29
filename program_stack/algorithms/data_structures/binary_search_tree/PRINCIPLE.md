# Binary Search Tree — Principle

## 1. Problem statement

Store comparable keys so that search, insert, and sorted enumeration are
efficient on average, using a binary tree ordered by key.

## 2. Idea & invariants

For every node `u`:

- all keys in `u.left` are `< u.key`
- all keys in `u.right` are `> u.key`

Inorder traversal (left, visit, right) therefore yields sorted order.

## 3. Correctness sketch

- `insert` walks left/right by comparison until a null child, then attaches —
  preserving the BST invariant at the new leaf.
- `contains` follows the same comparisons; if a node equals the key, success;
  if a null is reached, absence.
- Inductive argument: an empty tree is a BST; inserting a leaf into a BST
  yields a BST.

## 4. Complexity

| Quantity | Cost |
| -------- | ---- |
| Time (balanced / average) | `O(log n)` per op |
| Time (worst, skewed) | `O(n)` |
| Inorder of all nodes | `O(n)` |
| Extra space | `O(n)` nodes; recursion depth `O(h)` |

## 5. Implementation notes

Unbalanced BSTs degrade to lists on sorted inserts. Self-balancing trees
(AVL, red–black) restore `O(log n)` height. Iterative insert/search avoids
deep recursion on skewed trees.
