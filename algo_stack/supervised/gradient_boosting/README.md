# Gradient Boosting

> Stagewise additive ensemble of shallow regression trees. Squared loss for
> regression; binary logistic loss for classification (sigmoid probabilities).

## Install / import

```python
from algo_stack.supervised.gradient_boosting import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `GradientBoostingRegressor(*, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2, random_state=None)` | constructor | Squared-loss boosting. |
| `GradientBoostingClassifier(*, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2, random_state=None)` | constructor | Binary logistic boosting. |
| `.fit(X, y)` | method | Fit sequential residual trees. |
| `.predict(X)` | method | Labels or continuous values. |
| `.predict_proba(X)` | method | (classifier) Sigmoid of raw scores. |
| `.decision_function(X)` | method | (classifier) Raw additive scores. |
| `.init_` | fitted attr | Initial constant (mean or log-odds). |
| `.estimators_` | fitted attr | List of `DecisionTreeRegressor` stages. |

## Quick start

```python
import numpy as np
from algo_stack.supervised.gradient_boosting import GradientBoostingRegressor

rng = np.random.default_rng(0)
X = rng.uniform(-np.pi, np.pi, size=(200, 1))
y = np.sin(X.ravel()) + 0.15 * rng.normal(size=200)
reg = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=0).fit(X, y)
print(reg.score(X, y))
```

## See also

- Trees: `algo_stack.supervised.decision_tree`.
- `PRINCIPLE.md`, `EXTENSION.md`, `example.py`.
