# Random Forest

> Ensemble of CART trees trained on bootstrap samples with feature bagging.
> Soft-vote average for classification; mean of tree outputs for regression.

## Install / import

```python
from algo_stack.supervised.random_forest import (
    RandomForestClassifier,
    RandomForestRegressor,
)
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `RandomForestClassifier(*, n_estimators=100, max_depth=None, min_samples_split=2, max_features="sqrt", random_state=None)` | constructor | Classification forest. |
| `RandomForestRegressor(*, n_estimators=100, max_depth=None, min_samples_split=2, max_features="sqrt", random_state=None)` | constructor | Regression forest. |
| `.fit(X, y)` | method | Grow bootstrap trees. |
| `.predict(X)` / `.predict_proba(X)` | method | Averaged predictions. |
| `.estimators_` | fitted attr | List of fitted decision trees. |
| `.classes_` | fitted attr | (classifier) Sorted labels. |

## Quick start

```python
from algo_stack.supervised.random_forest import RandomForestClassifier
import numpy as np

rng = np.random.default_rng(0)
X = np.vstack([rng.normal([-2, 0], 0.8, (50, 2)), rng.normal([2, 0], 0.8, (50, 2))])
y = np.repeat([0, 1], 50)
clf = RandomForestClassifier(n_estimators=20, max_depth=4, random_state=0).fit(X, y)
print(clf.score(X, y))
```

## See also

- Reuses `algo_stack.supervised.decision_tree`.
- `PRINCIPLE.md`, `EXTENSION.md`, `example.py`.
