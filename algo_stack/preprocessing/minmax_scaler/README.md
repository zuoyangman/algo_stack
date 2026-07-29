# MinMaxScaler

> Scale each feature into a fixed range (default `[0, 1]`).

## Install / import

```python
from algo_stack.preprocessing.minmax_scaler import MinMaxScaler
# or
from algo_stack.preprocessing import MinMaxScaler
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `MinMaxScaler(*, feature_range=(0, 1))` | constructor | Target `(min, max)`. |
| `.fit(X)` / `.transform(X)` / `.inverse_transform(X)` / `.fit_transform(X)` | method | Learn range, apply / undo. |
| `.min_`, `.scale_`, `.data_min_`, `.data_max_` | fitted attr | Affine map parameters. |

## Quick start

```python
import numpy as np
from algo_stack.preprocessing import MinMaxScaler

X = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
Z = MinMaxScaler().fit_transform(X)  # columns in [0, 1]
```

## Run the bundled example

```bash
python -m algo_stack.preprocessing.minmax_scaler.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
