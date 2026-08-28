# <Algorithm Name>

> One-sentence description of what this algorithm does and when to use it.

## Install / import

```python
from algo_stack.<category>.<algorithm_name> import <ClassName>
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `<ClassName>(...)` | constructor | List every keyword-only hyperparameter and its default. |
| `.fit(X, y)`       | method      | Fit the model. Returns `self`. |
| `.predict(X)`      | method      | Predict on new data. |
| `.score(X, y)`     | method      | Default score (R² for regressors, accuracy for classifiers). |
| `.<attr>_`         | fitted attr | Describe each fitted attribute (e.g. `coef_`, `intercept_`). |

## Quick start

```python
import numpy as np
from algo_stack.<category>.<algorithm_name> import <ClassName>

X = ...  # shape (n_samples, n_features)
y = ...  # shape (n_samples,)

model = <ClassName>().fit(X, y)
print(model.predict(X[:5]))
print("score =", model.score(X, y))
```

## Run the bundled example

```bash
python -m algo_stack.<category>.<algorithm_name>.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
