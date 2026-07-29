# Binary Search Tree — Extension Guide

## 1. Extension surface

Core: `insert`, `contains`, `inorder`. Optional: `min`, `max`, `delete`,
`height`.

## 2. Common variants

### 2.1 Delete
Three cases: leaf, one child, two children (replace with inorder successor).

### 2.2 AVL / red–black
Rebalance after insert/delete to guarantee logarithmic height.

### 2.3 Order-statistic tree
Augment nodes with subtree sizes for rank/select queries.

## 3. Invariants to preserve

- `main` exits 0 only on success.
- Success line: `binary_search_tree: ok`
- Cross-language API naming stays mappable.

## 4. Pitfalls

- Treating duplicates inconsistently across languages.
- Breaking the BST invariant by inserting on the wrong side.
- Stack overflow on deep recursion with sorted input.

## 5. Suggested follow-up algorithms

- Heap / priority queue
- AVL tree
- Treap
