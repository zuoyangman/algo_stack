# Ridge Regression

> Ordinary-least-squares linear regression with an L2 penalty on the weights.
> Closed-form solution via the regularised normal equations; intercept is not
> penalised.

## Install / import

```python
from algo_stack.supervised.ridge import Ridge
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `Ridge(*, alpha=1.0, fit_intercept=True)` | constructor | `alpha` is the L2 strength (≥ 0). |
| `.fit(X, y)`   | method      | Estimate weights. Returns `self`. |
| `.predict(X)`  | method      | Return predictions of shape `(n_samples,)`. |
| `.score(X, y)` | method      | Coefficient of determination, R². |
| `.coef_`       | fitted attr | Weights, shape `(n_features,)`. |
| `.intercept_`  | fitted attr | Scalar bias term (0.0 if `fit_intercept=False`). |

## Quick start

```python
import numpy as np
from algo_stack.supervised.ridge import Ridge

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 5))
y = X @ np.array([1.5, -2.0, 0.5, 0.0, 0.0]) + 3.0 + 0.1 * rng.normal(size=200)

model = Ridge(alpha=1.0).fit(X, y)
print(model.coef_, model.intercept_)
print("R² =", model.score(X, y))
```

## Run the bundled example

```bash
python -m algo_stack.supervised.ridge.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
