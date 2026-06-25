# Logistic Regression

> Linear classifier with sigmoid (binary) or softmax (multinomial) output,
> trained by full-batch gradient descent with optional L2 penalty.

## Install / import

```python
from algo_stack.supervised.logistic_regression import LogisticRegression
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `LogisticRegression(*, learning_rate=0.1, n_iter=1000, l2=0.0, tol=1e-6, fit_intercept=True, random_state=None)` | constructor | All hyperparameters keyword-only. |
| `.fit(X, y)`            | method      | Trains. Auto-detects binary vs. multinomial from `np.unique(y)`. Returns `self`. |
| `.predict(X)`           | method      | Returns class labels of shape `(n_samples,)`. |
| `.predict_proba(X)`     | method      | Returns class probabilities of shape `(n_samples, n_classes)`. |
| `.decision_function(X)` | method      | Returns raw logits. |
| `.score(X, y)`          | method      | Accuracy. |
| `.classes_`             | fitted attr | Class label vocabulary. |
| `.coef_`                | fitted attr | Shape `(n_features,)` (binary) or `(n_classes, n_features)` (multinomial). |
| `.intercept_`           | fitted attr | `float` (binary) or shape `(n_classes,)` (multinomial). |
| `.n_iter_`              | fitted attr | Iterations actually run. |
| `.loss_history_`        | fitted attr | List of per-iteration mean log-losses. |

## Quick start

```python
import numpy as np
from algo_stack.supervised.logistic_regression import LogisticRegression

rng = np.random.default_rng(0)
X = np.vstack([rng.normal([-2, 0], size=(100, 2)),
               rng.normal([+2, 0], size=(100, 2))])
y = np.array([0] * 100 + [1] * 100)

clf = LogisticRegression(learning_rate=0.5, n_iter=500).fit(X, y)
print("acc =", clf.score(X, y))
print("p(y=1|x) =", clf.predict_proba(X[:3])[:, 1])
```

## Run the bundled example

```bash
python -m algo_stack.supervised.logistic_regression.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
