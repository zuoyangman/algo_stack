# K-Nearest Neighbours (KNN)

> Non-parametric instance-based learner: classify / regress a query point by
> looking up its `k` nearest training samples and aggregating their labels.

This module exposes both a classifier and a regressor variant that share the
same neighbour-lookup code.

## Install / import

```python
from algo_stack.supervised.knn import KNNClassifier, KNNRegressor
```

## Public API

Both classes share the constructor signature:

| Name | Type | Description |
| ---- | ---- | ----------- |
| `KNNClassifier(*, n_neighbors=5, weights="uniform", metric="euclidean")` | constructor | `weights` is `"uniform"` or `"distance"`. `metric` currently only supports `"euclidean"`. |
| `KNNRegressor (*, n_neighbors=5, weights="uniform", metric="euclidean")` | constructor | Same. |
| `.fit(X, y)`            | method      | Just memorises the training set. Returns `self`. |
| `.predict(X)`           | method      | Labels (classifier) or values (regressor). |
| `.predict_proba(X)`     | method      | (classifier only) Class probabilities. |
| `.score(X, y)`          | method      | Accuracy (classifier) or R² (regressor). |
| `.classes_`             | fitted attr | (classifier) sorted class labels. |
| `.X_train_`, `.y_train_` | fitted attr | Stored training data. |

## Quick start

```python
import numpy as np
from algo_stack.supervised.knn import KNNClassifier

X = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]])
y = np.array([0, 0, 1, 1])
print(KNNClassifier(n_neighbors=1).fit(X, y).predict([[0.5, 0.0]]))   # -> [0]
```

## Run the bundled example

```bash
python -m algo_stack.supervised.knn.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
