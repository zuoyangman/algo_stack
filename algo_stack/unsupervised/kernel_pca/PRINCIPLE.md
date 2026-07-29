# Kernel PCA — Principle

## 1. Problem statement

Perform PCA in feature space `φ(x)` using only kernel evaluations `K(x, x′)`.

## 2. Model

Build `K`, centre `K̃ = HKH`, solve `K̃ v = λ v`, project with
`y = K̃_{*·} (v / √λ)`.

## 3. Algorithm

```
K ← kernel(X, X); centre
eigh → top positive eigenvalues / vectors
α ← v / √λ
```

## 4. Complexity

`O(n² d)` for Gram + `O(n³)` for eigh.

## 5. Notes

Out-of-sample points are centred with training row/global means.
