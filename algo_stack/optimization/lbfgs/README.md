# L-BFGS (Limited-memory BFGS)

> Quasi-Newton optimiser with strong Wolfe line search — the workhorse behind
> many sklearn linear models (`solver="lbfgs"`).

## Install / import

```python
from algo_stack.optimization.lbfgs import minimize_lbfgs, LBFGS, wolfe_line_search
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `minimize_lbfgs(fun, x0, *, jac, max_iter=200, m=10, tol=1e-6, c1=1e-4, c2=0.9)` | function | Minimise scalar `fun` given gradient `jac`. Returns `LBFGSResult`. |
| `wolfe_line_search(fun, grad, x, p, ...)` | function | Strong Wolfe line search along direction `p`. |
| `LBFGS(*, max_iter, m, tol, c1, c2)` | class | Thin wrapper; call `.minimize(fun, x0, jac)`. |
| `LBFGSResult` | dataclass | `x`, `fun`, `grad`, `n_iter`, `n_fev`, `n_gev`, `success`, `message`, `loss_curve`. |

## Quick start

```python
import numpy as np
from algo_stack.optimization.lbfgs import minimize_lbfgs

def fun(z):
    return float(np.sum(z ** 2))

def jac(z):
    return 2 * z

res = minimize_lbfgs(fun, np.ones(5), jac=jac)
print(res.x, res.fun, res.success)
```

## Run the bundled example

```bash
python -m algo_stack.optimization.lbfgs.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
