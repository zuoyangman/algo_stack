# Softmax Classifier

> The standard linear classification head in deep learning: `scores = Wx + b`
> followed by softmax and cross-entropy loss, trained with mini-batch SGD.

## Install / import

```python
from algo_stack.supervised.softmax_classifier import SoftmaxClassifier
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `SoftmaxClassifier(*, learning_rate=0.1, optimizer="momentum", n_epochs=200, batch_size=32, l2=0.0, tol=1e-5, shuffle=True, random_state=None, fit_intercept=True)` | constructor | `optimizer` is `"sgd"`, `"momentum"`, or `"adam"`. |
| `.fit(X, y)` | method | Mini-batch training. Returns `self`. |
| `.predict(X)` / `.predict_proba(X)` / `.decision_function(X)` | methods | Standard classifier API. |
| `.score(X, y)` | method | Accuracy. |
| `.coef_` | fitted attr | Shape `(n_classes, n_features)`. |
| `.intercept_` | fitted attr | Shape `(n_classes,)`. |
| `.loss_curve_` | fitted attr | Per-epoch cross-entropy. |
| `.n_iter_` | fitted attr | Epochs run. |

## Quick start

```python
import numpy as np
from algo_stack.supervised.softmax_classifier import SoftmaxClassifier

rng = np.random.default_rng(0)
X = np.vstack([rng.normal([-2, 0], 0.5, (100, 2)),
               rng.normal([+2, 0], 0.5, (100, 2))])
y = np.array([0] * 100 + [1] * 100)

clf = SoftmaxClassifier(optimizer="adam", n_epochs=200).fit(X, y)
print("acc =", clf.score(X, y))
```

## Run the bundled example

```bash
python -m algo_stack.supervised.softmax_classifier.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
