# StandardScaler

> Centre and scale features to zero mean and unit variance.

## Install / import

```python
from algo_stack.preprocessing.standard_scaler import StandardScaler
# or
from algo_stack.preprocessing import StandardScaler
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `StandardScaler(*, with_mean=True, with_std=True)` | constructor | Toggle centring / scaling. |
| `.fit(X)` / `.transform(X)` / `.inverse_transform(X)` / `.fit_transform(X)` | method | Learn stats, apply / undo scaling. |
| `.mean_`, `.scale_` | fitted attr | Per-feature mean and scale. |

## Quick start

```python
import numpy as np
from algo_stack.preprocessing import StandardScaler

X = np.random.randn(100, 3) * 5 + 10
Z = StandardScaler().fit_transform(X)
```

## Run the bundled example

```bash
python -m algo_stack.preprocessing.standard_scaler.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
