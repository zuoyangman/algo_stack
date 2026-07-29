# Decision Tree (CART)

> Binary CART trees: Gini impurity for classification, MSE for regression.
> Axis-aligned threshold splits; optional feature bagging via `max_features`.

## Install / import

```python
from algo_stack.supervised.decision_tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `DecisionTreeClassifier(*, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=None)` | constructor | Gini CART classifier. |
| `DecisionTreeRegressor(*, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=None)` | constructor | MSE CART regressor. |
| `.fit(X, y)` | method | Grow `tree_`. Returns `self`. |
| `.predict(X)` | method | Labels or continuous values. |
| `.predict_proba(X)` | method | (classifier) Class probabilities from leaf class fractions. |
| `.score(X, y)` | method | Accuracy or R². |
| `.tree_` | fitted attr | Root `Node` (`feature`, `threshold`, `left`, `right`, `value`, `n_samples`). |
| `.classes_` | fitted attr | (classifier) Sorted class labels. |

## Quick start

```python
import numpy as np
from algo_stack.supervised.decision_tree import DecisionTreeClassifier

# XOR needs depth >= 2
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 1, 1, 0])
clf = DecisionTreeClassifier(max_depth=2).fit(X, y)
print(clf.predict(X))  # [0, 1, 1, 0]
```

## See also

- `PRINCIPLE.md` — Gini / MSE split criterion and recursion.
- `EXTENSION.md` — pruning, multiway splits, missing values.
- `example.py` — runnable demo.
