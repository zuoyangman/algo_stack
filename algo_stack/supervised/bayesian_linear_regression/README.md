# Bayesian Linear Regression

> Linear regression with an isotropic Gaussian prior on the weights. Yields a
> closed-form Gaussian posterior and a predictive distribution
> (`predict` = mean, `predict_std` = std).

## Install / import

```python
from algo_stack.supervised.bayesian_linear_regression import BayesianLinearRegression
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `BayesianLinearRegression(*, alpha=1.0, beta=1.0, fit_intercept=True)` | constructor | `alpha` = prior precision, `beta` = noise precision. |
| `.fit(X, y)` | method | Compute posterior mean / covariance. |
| `.predict(X)` | method | Predictive mean, shape `(n_samples,)`. |
| `.predict_std(X)` | method | Predictive std, shape `(n_samples,)`. |
| `.predict_var(X)` | method | Predictive variance. |
| `.score(X, y)` | method | R² on the predictive mean. |
| `.coef_` / `.intercept_` | fitted | Posterior mean weights / bias. |
| `.alpha_` / `.beta_` | fitted | Precisions used at fit time. |
| `.sigma_` | fitted | Posterior weight covariance. |

## Quick start

```python
import numpy as np
from algo_stack.supervised.bayesian_linear_regression import BayesianLinearRegression

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 3))
y = X @ np.array([1.0, -2.0, 0.5]) + 0.2 * rng.normal(size=100)

model = BayesianLinearRegression(alpha=1.0, beta=25.0).fit(X, y)
print(model.coef_, model.predict_std(X)[:5])
```

## Run the bundled example

```bash
python -m algo_stack.supervised.bayesian_linear_regression.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
