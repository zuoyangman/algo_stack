# OneHotEncoder — Extension Guide

## 1. Extension surface

- `drop` — `None` or `'first'`; could add dropping a named category.
- Category order is first-seen; could switch to sorted order.

## 2. Variants

- Ordinal encoding for ordered categories.
- Target / frequency encoding for high-cardinality features.
- Sparse CSR output for very wide encodings.

## 3. Invariants

- Each fitted input row activates exactly one column per feature block
  (zero if that category was dropped or is unknown).
- `transform` output dtype is `float64`.

## 4. Pitfalls

- High-cardinality columns explode dimensionality.
- Always `fit` on training categories only; new labels become zeros.
