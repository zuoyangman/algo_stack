# Linear Regression — Extension Guide

## 1. Extension surface

Designed-to-be-modified parts of `linear_regression.py`:

- **Constructor arguments**: add new keyword-only hyperparameters
  (`alpha`, `l1_ratio`, `weights`, `kernel`, …). Always keep them
  keyword-only and store them unchanged in `__init__`.
- **`solver` dispatch block in `fit`**: a single `if/elif` switch.
  New solvers should be added as a new branch *and* documented in the
  docstring's `Parameters` section.
- **`predict`**: do not change its public signature; if a variant needs more
  output (e.g. predictive variance), expose it via an additional method like
  `predict_var(X)`.

## 2. Common variants and how to add them

### 2.1 Ridge Regression (L2)

Add `alpha: float = 0.0` to `__init__`. In the `"normal"` branch replace

```
theta = np.linalg.solve(XtX + alpha * I_modified, Xty)
```

where `I_modified` has zero in the row/column corresponding to the intercept
column so that the bias is *not* penalised. Recommended to expose this as a
sibling class `Ridge` rather than overloading `LinearRegression`, mirroring
sklearn's API; the math reuse is the only thing shared.

### 2.2 Lasso (L1) / ElasticNet

L1 has no closed form. Add a `"cgd"` (coordinate gradient descent) solver
branch in `fit`. The convergence loop, soft-threshold function and
`max_iter` / `tol` hyperparameters should live in a private function
`_coordinate_descent` next to `LinearRegression`. Keep the public
`LinearRegression` class L2-free; create a `Lasso` class that *delegates* to
the same input validation and `predict` code.

### 2.3 Weighted Least Squares

Add `sample_weight: np.ndarray | None = None` to `fit`. Multiply both
`X̃` and `y` by `sqrt(sample_weight)` before solving; the resulting normal
equations are equivalent.

### 2.4 Polynomial Regression

This is *feature engineering*, not a new algorithm: implement a separate
`PolynomialFeatures` transformer under `algo_stack/preprocessing/` and
compose: `LinearRegression().fit(PolynomialFeatures(deg=3).fit_transform(X), y)`.

## 3. Invariants the implementation relies on

- `fit` returns `self` and never mutates `X` or `y` in place.
- After `fit`, `self.coef_` has shape `(n_features,)` and `self.intercept_` is
  a Python `float`. Variants targeting multi-output regression must change
  this contract explicitly (and bump the doc).
- `predict` always returns shape `(n_samples,)` for the single-output case.

## 4. Pitfalls

- Don't compute `np.linalg.inv(XᵀX)` — always use `solve` or `lstsq`.
  Explicit inversion is both slower and less stable.
- For very wide `X` (`d > n`) the `"normal"` solver becomes singular; prefer
  `"lstsq"` or move to a regularised variant.
- Be careful with the intercept when adding regularisation: the bias column
  should generally *not* be penalised.

## 5. Suggested follow-up algorithms

- Ridge / Lasso / ElasticNet (see §2)
- Bayesian Linear Regression (predictive distribution)
- Generalised Linear Models (link functions → Logistic, Poisson, …)
