# Perceptron

> The original threshold neuron (Rosenblatt, 1957): a linear classifier trained
> by the mistake-driven perceptron learning rule.

## Install / import

```python
from algo_stack.supervised.perceptron import Perceptron
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `Perceptron(*, learning_rate=1.0, n_iter=1000, tol=0.0, shuffle=True, random_state=None, fit_intercept=True)` | constructor | All keyword-only. |
| `.fit(X, y)` | method | Online / epoch-wise training. Returns `self`. |
| `.predict(X)` | method | Hard threshold predictions. |
| `.score(X, y)` | method | Accuracy. |
| `.classes_` | fitted attr | Sorted class labels. |
| `.coef_` | fitted attr | `(n_features,)` for binary; `(n_classes, n_features)` for multiclass OvR. |
| `.intercept_` | fitted attr | `float` (binary) or `(n_classes,)` (multiclass). |
| `.n_iter_` | fitted attr | Epochs actually run. |
| `.mistakes_history_` | fitted attr | Mistake count per epoch. |

## Quick start

```python
import numpy as np
from algo_stack.supervised.perceptron import Perceptron

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 1, 1])  # AND gate — linearly separable
clf = Perceptron(learning_rate=1.0, n_iter=20).fit(X, y)
print(clf.predict(X))
```

## Run the bundled example

```bash
python -m algo_stack.supervised.perceptron.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
