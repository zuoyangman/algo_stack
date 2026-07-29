# SVM — Principle

## 1. Problem statement

Find a maximum-margin separator in a (possibly infinite-dimensional) feature
space induced by a kernel `K`, allowing soft-margin slack controlled by `C`.

- **Classification (SVC):** soft-margin C-SVM, binary SMO + One-vs-Rest.
- **Regression (SVR):** Kernel Ridge dual (squared loss), not ε-SVR.

## 2. Mathematical model (C-SVM)

Primal (with feature map `φ`):

```
min_{w,b,ξ}  ½‖w‖² + C Σ_i ξ_i
s.t.  y_i (⟨w, φ(x_i)⟩ + b) ≥ 1 − ξ_i ,  ξ_i ≥ 0 ,  y_i ∈ {−1,+1}
```

Dual:

```
max_α  Σ_i α_i − ½ Σ_{i,j} α_i α_j y_i y_j K(x_i, x_j)
s.t.   0 ≤ α_i ≤ C ,  Σ_i α_i y_i = 0
```

Decision: `f(x) = Σ_i α_i y_i K(x_i, x) + b`. Support vectors have `α_i > 0`.

## 3. SMO (binary)

Sequential Minimal Optimisation (Platt) optimises **two** dual variables at a
time so the linear equality constraint stays satisfied.

For a pair `(i1, i2)`:

1. Compute error `E_i = f(x_i) − y_i`.
2. Bound `α₂` into `[L, H]` from the box + equality constraints.
3. Unconstrained step: `α₂ ← α₂ + y₂ (E₁ − E₂) / η` with
   `η = K₁₁ + K₂₂ − 2 K₁₂`, then clip to `[L, H]`.
4. Update `α₁` to keep `Σ α y = 0`, refresh the cached `f`, recompute `b`
   from unbound support vectors (`0 < α < C`).

Outer loop: full pass over all examples, then passes over unbound SVs,
until no KKT violations remain (within `tol`) or `max_iter` is hit.

## 4. Multiclass

One-vs-Rest: train `K` binary SVMs; predict `argmax_c f_c(x)`.

## 5. SVR — Kernel Ridge dual (design choice)

Instead of ε-insensitive SMO we solve

```
α = (K + λ I)^{-1} y ,   λ = 1/C ,
ŷ(x) = K(x, X_train) α + b
```

This is Kernel Ridge Regression in the dual. Reasons:

- Closed form, exact for squared loss — ideal for didactic code.
- Same kernel machinery as SVC (`linear` / `rbf`).
- Dense dual: every training point is a "support vector".

True ε-SVR SMO is sketched in EXTENSION.md.

## 6. Kernels

| Kernel | Formula |
| ------ | ------- |
| linear | `⟨x, z⟩` |
| rbf    | `exp(−γ ‖x − z‖²)` with `γ = "scale"` → `1/(n_features · Var(X))` |

## 7. Complexity

| Quantity | Cost |
| -------- | ---- |
| Kernel matrix | `O(n² d)` |
| SMO (typical) | roughly `O(n²)`–`O(n³)` depending on sparsity / iterations |
| Memory | `O(n²)` for the Gram matrix |
| SVR solve | `O(n³)` for the linear system |

## 8. Implementation notes

- Labels are mapped to `{−1,+1}` internally; `classes_` stores originals.
- Bias uses the average over unbound SVs: `b = mean(y_i − Σ_j α_j y_j K_ij)`.
- Shared kernel helpers live in `algo_stack.utils.kernels`.
