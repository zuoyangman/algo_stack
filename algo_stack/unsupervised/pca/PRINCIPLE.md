# PCA — Principle

## 1. Problem statement

Find orthonormal directions maximising captured variance of centred `X`.

## 2. Model

Centre `X_c = X − μ`. SVD `X_c = U S Vᵀ`. Rows of `Vᵀ` are principal components;
explained variance is `S² / (n−1)`.

## 3. Algorithm

```
μ ← mean(X)
U, S, Vt ← svd(X − μ)
keep top n_components rows of Vt
```

## 4. Complexity

`O(min(n d², d n²))` via economy SVD.

## 5. Notes

`transform` is `(X−μ) components_ᵀ`; `inverse_transform` is `Z components_ + μ`.
