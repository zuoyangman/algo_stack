# Lasso Regression

> Ordinary-least-squares linear regression with an L1 penalty on the weights,
> solved by cyclic coordinate descent with soft-thresholding. Produces sparse
> solutions; intercept is not penalised.

## Install / import

```python
from algo_stack.supervised.lasso import Lasso
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `Lasso(*, alpha=1.0, max_iter=1000, tol=1e-4, fit_intercept=True)` | constructor | `alpha` is the L1 strength (≥ 0). |
| `.fit(X, y)`   | method      | Estimate sparse weights. Returns `self`. |
| `.predict(X)`  | method      | Return predictions of shape `(n_samples,)`. |
| `.score(X, y)` | method      | Coefficient of determination, R². |
| `.coef_`       | fitted attr | Weights, shape `(n_features,)` (often sparse). |
| `.intercept_`  | fitted attr | Scalar bias term. |
| `.n_iter_`     | fitted attr | Coordinate sweeps actually run. |

## Quick start

```python
import numpy as np
from algo_stack.supervised.lasso import Lasso

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 10))
w_true = np.zeros(10); w_true[:3] = [2.0, -1.5, 1.0]
y = X @ w_true + 0.1 * rng.normal(size=200)

model = Lasso(alpha=0.1).fit(X, y)
print(model.coef_)
```

## Run the bundled example

```bash
python -m algo_stack.supervised.lasso.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
