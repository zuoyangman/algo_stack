# PolynomialFeatures

> Expand a feature matrix into polynomial and interaction terms.

## Install / import

```python
from algo_stack.preprocessing.polynomial_features import PolynomialFeatures
# or
from algo_stack.preprocessing import PolynomialFeatures
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `PolynomialFeatures(*, degree=2, include_bias=True, interaction_only=False)` | constructor | Expansion knobs. |
| `.fit(X)` / `.transform(X)` / `.fit_transform(X)` | method | Build basis; apply. |
| `.n_features_in_`, `.n_output_features_`, `.powers_` | fitted attr | Input/output sizes and exponent matrix. |

## Quick start

```python
import numpy as np
from algo_stack.preprocessing import PolynomialFeatures

X = np.array([[2.0, 3.0]])
Z = PolynomialFeatures(degree=2).fit_transform(X)
# → [[1, 2, 3, 4, 6, 9]]
```

## Run the bundled example

```bash
python -m algo_stack.preprocessing.polynomial_features.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
