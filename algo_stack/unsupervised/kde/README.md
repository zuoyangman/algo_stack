# Kernel Density Estimation

> Non-parametric density estimate with a fixed-bandwidth Gaussian kernel.

## Install / import

```python
from algo_stack.unsupervised.kde import KernelDensity
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `KernelDensity(*, bandwidth=1.0, kernel="gaussian")` | constructor | Keyword-only. |
| `.fit(X)` / `.score_samples(X)` / `.score(X)` | method | Store training points; log-density; mean log-likelihood. |
| `.X_fit_`, `.n_features_in_` | fitted attr | Training data copy. |

## Quick start

See `example.py`.

## Run the bundled example

```bash
python -m algo_stack.unsupervised.kde.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
