# Linear Regression — Principle

## 1. Problem statement

Given a training set of `n` samples `{(x_i, y_i)}` with `x_i ∈ ℝᵈ` and
`y_i ∈ ℝ`, assume

```
y_i = w · x_i + b + ε_i,   ε_i ~ N(0, σ²) i.i.d.
```

We want to estimate `(w, b)` that maximise the likelihood, which under the
Gaussian noise assumption is equivalent to minimising the sum of squared
residuals.

## 2. Mathematical model

Stack the samples into a design matrix `X ∈ ℝ^{n×d}` and target vector
`y ∈ ℝⁿ`. Augment `X` with a leading/trailing column of ones to absorb the
intercept into `θ = [w; b]`, so the model is `ŷ = X̃ θ`.

The objective is

```
L(θ) = ||X̃ θ − y||² = (X̃ θ − y)ᵀ (X̃ θ − y).
```

## 3. Derivation

`L` is convex and quadratic in `θ`. Setting `∇θ L = 0`:

```
∇θ L = 2 X̃ᵀ (X̃ θ − y) = 0
   ⇒  X̃ᵀ X̃ θ* = X̃ᵀ y.        (the "normal equations")
```

If `X̃ᵀ X̃` is invertible the unique minimiser is

```
θ* = (X̃ᵀ X̃)⁻¹ X̃ᵀ y.
```

When `X̃` is rank-deficient or ill-conditioned, prefer the minimum-norm
least-squares solution given by the pseudo-inverse / SVD, which is exactly
what `numpy.linalg.lstsq` returns.

## 4. Algorithm

```
1. Validate inputs (shape, finiteness).
2. If fit_intercept: append a column of ones to X to form X̃, else X̃ = X.
3. solver == "lstsq":  θ, _, rank, _ = np.linalg.lstsq(X̃, y, rcond=None)
   solver == "normal": θ = np.linalg.solve(X̃ᵀX̃, X̃ᵀy)
4. Split θ into (coef_, intercept_).
```

## 5. Complexity

| Quantity | Cost |
| -------- | ---- |
| Training time   | `O(n d²)` for normal equations / lstsq via SVD. |
| Training memory | `O(n d)` for the design matrix copy. |
| Prediction time | `O(d)` per sample. |

## 6. Numerical & implementation notes

- We default to `"lstsq"` because it uses an SVD-based pseudo-inverse and
  handles rank-deficient `X` gracefully; the `"normal"` solver is faster
  but squares the condition number of `X` and can blow up on ill-conditioned
  inputs.
- The intercept is absorbed into `θ` by augmenting `X`. This is equivalent
  to centring `X` and `y` and recovering `b` afterwards; both give the same
  answer up to floating-point noise.
- We do **not** centre/standardise features by default — that responsibility
  belongs to the user via `algo_stack.utils.StandardScaler` (or by setting
  `fit_intercept=False` after they have centred manually).
