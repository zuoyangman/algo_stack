# Kernel Ridge — Principle

## 1. Problem statement

Learn a function in a reproducing kernel Hilbert space (RKHS) with squared
loss and an RKHS-norm penalty:

```
min_f  Σ_i (f(x_i) − y_i)² + λ ‖f‖²_H
```

By the representer theorem, the optimum lies in the span of training kernels:
`f(x) = Σ_i α_i K(x_i, x)` (+ optional intercept).

## 2. Closed form

Let `K_{ij} = K(x_i, x_j)`. Substituting the representer expansion yields

```
α = (K + λ I)^{-1} y
ŷ(x) = k(x)^T α          where  k(x)_i = K(x_i, x)
```

With `fit_intercept=True` we centre `y` before solving and set
`b = mean(y) − mean(K α)` so training residuals have zero mean.

## 3. Kernels

| Name | Formula |
| ---- | ------- |
| linear | `⟨x, z⟩` |
| rbf | `exp(−γ ‖x − z‖²)` |
| polynomial | `(γ ⟨x, z⟩ + coef0)^{degree}` |

`gamma="scale"` → `1 / (n_features · Var(X))` (sklearn convention).

## 4. Complexity

| Quantity | Cost |
| -------- | ---- |
| Build `K` | `O(n² d)` |
| Solve `(K + λI) α = y` | `O(n³)` |
| Predict `m` points | `O(m n d)` kernel + `O(m n)` matvec |
| Memory | `O(n²)` |

## 5. Relation to SVR in this repo

`algo_stack.supervised.svm.SVR` uses the **same** dual closed form with
`λ = 1/C`. KernelRidge exposes `alpha=λ` directly and also supports a
polynomial kernel.

## 6. Numerical notes

- `λ > 0` guarantees positive-definiteness even if `K` is singular.
- We use `np.linalg.solve` (Cholesky under the hood for SPD systems).
- Shared helpers: `algo_stack.utils.kernels`.
