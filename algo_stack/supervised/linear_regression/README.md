# Linear Regression

> Ordinary-least-squares linear regression: fit a hyperplane that minimises the
> sum of squared residuals between predictions and targets.

## Install / import

```python
from algo_stack.supervised.linear_regression import LinearRegression
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `LinearRegression(*, fit_intercept=True, solver="lstsq")` | constructor | `solver` is `"lstsq"` (default, via `numpy.linalg.lstsq`) or `"normal"` (normal equations). |
| `.fit(X, y)`   | method      | Estimate weights. Returns `self`. |
| `.predict(X)`  | method      | Return predictions of shape `(n_samples,)`. |
| `.score(X, y)` | method      | Coefficient of determination, R². |
| `.coef_`       | fitted attr | Weights, shape `(n_features,)`. |
| `.intercept_`  | fitted attr | Scalar bias term (0.0 if `fit_intercept=False`). |
| `.rank_`       | fitted attr | Effective rank of the design matrix. |

## Quick start

```python
import numpy as np
from algo_stack.supervised.linear_regression import LinearRegression

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3))
y = X @ np.array([1.5, -2.0, 0.5]) + 4.0 + 0.1 * rng.normal(size=200)

model = LinearRegression().fit(X, y)
print(model.coef_, model.intercept_)
print("R² =", model.score(X, y))
```

## Run the bundled example

```bash
python -m algo_stack.supervised.linear_regression.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
