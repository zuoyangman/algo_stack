# MinMaxScaler — Principle

## 1. Problem statement

Map each feature from its observed interval `[data_min_j, data_max_j]` onto
a user-chosen interval `[a, b]` (default `[0, 1]`).

## 2. Model

```
scale_j = (b − a) / (data_max_j − data_min_j)
min_j   = a − data_min_j * scale_j
x'_ij   = x_ij * scale_j + min_j
```

Zero-range features use `data_range = 1` so they map to the constant `a`.

## 3. Algorithm

```
data_min ← min(X, axis=0)
data_max ← max(X, axis=0)
range    ← data_max − data_min;  range[range==0] ← 1
scale    ← (b − a) / range
min      ← a − data_min * scale
```

## 4. Complexity

`O(n d)` fit / transform; `O(d)` fitted state.

## 5. Notes

Affine and invertible (when ranges are positive): `inverse_transform` undoes
the map exactly up to floating-point noise.
