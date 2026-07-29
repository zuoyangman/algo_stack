# OneHotEncoder

> Encode categorical columns as dense binary indicator features.

## Install / import

```python
from algo_stack.preprocessing.one_hot_encoder import OneHotEncoder
# or
from algo_stack.preprocessing import OneHotEncoder
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `OneHotEncoder(*, drop=None)` | constructor | `drop='first'` drops first category per feature. |
| `.fit(X)` / `.transform(X)` / `.fit_transform(X)` | method | Learn categories; emit dense `float64` array. |
| `.categories_` | fitted attr | List of category arrays, one per input feature. |

`X` may be 1-D (single feature) or 2-D (one categorical column per feature).

## Quick start

```python
import numpy as np
from algo_stack.preprocessing import OneHotEncoder

X = np.array([["cat"], ["dog"], ["cat"]])
Z = OneHotEncoder().fit_transform(X)  # shape (3, 2)
```

## Run the bundled example

```bash
python -m algo_stack.preprocessing.one_hot_encoder.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
