# Kernel Ridge Regression

> Closed-form ridge regression in an RKHS: solve for dual coefficients
> `α = (K + λ I)^{-1} y` and predict `ŷ = K(x, X) α`.

## Install / import

```python
from algo_stack.supervised.kernel_ridge import KernelRidge
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `KernelRidge(*, alpha=1.0, kernel="rbf", gamma="scale", degree=3, coef0=1.0, fit_intercept=True)` | constructor | `alpha` is λ. `kernel` ∈ `{"linear","rbf","polynomial"}`. |
| `.fit(X, y)` | method | Builds `K`, solves the dual linear system. |
| `.predict(X)` | method | `K(X, X_fit_) @ dual_coef_ + intercept_`. |
| `.score(X, y)` | method | R². |
| `.dual_coef_` | fitted | Dual coefficients `α`. |
| `.X_fit_` | fitted | Training inputs. |
| `.intercept_` | fitted | Bias. |
| `.gamma_` | fitted | Resolved bandwidth. |

## Quick start

```python
import numpy as np
from algo_stack.supervised.kernel_ridge import KernelRidge

rng = np.random.default_rng(0)
X = rng.uniform(-2, 2, size=(80, 1))
y = np.sin(X.ravel())
reg = KernelRidge(alpha=0.1, kernel="rbf").fit(X, y)
print(reg.score(X, y))
```

## Run the bundled example

```bash
python -m algo_stack.supervised.kernel_ridge.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
