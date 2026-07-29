# OneHotEncoder — Principle

## 1. Problem statement

Most numeric learners cannot ingest raw labels (`"red"`, `"blue"`, …).
One-hot encoding replaces each category with a binary column.

## 2. Model

For feature `j` with categories `C_j = {c_1, …, c_K}`:

```
φ_k(x) = 1[x = c_k],  k = 1..K
```

With `drop='first'`, omit `c_1` (K−1 columns) to remove the linear dependence
among indicators.

## 3. Algorithm

```
for each column j:
    categories_j ← unique values in order of first appearance
    if drop == 'first': categories_j ← categories_j[1:]
transform: for each category, emit a 0/1 column; hstack blocks
```

Unknown categories at transform time yield an all-zero block for that feature.

## 4. Complexity

`O(n d K)` transform where `K` is categories per feature; output width is
`Σ |categories_j|`.

## 5. Notes

Output is always a dense `float64` array (NumPy-only, no sparse matrices).
