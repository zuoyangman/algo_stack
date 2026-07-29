# Multi-Layer Perceptron (MLP)

> Fully-connected feed-forward neural network trained by mini-batch SGD with
> momentum. NumPy-only reference implementation with manual backprop.

This module exposes:

- `MLPRegressor`  — identity output + MSE loss
- `MLPClassifier` — softmax output + cross-entropy loss

## Install / import

```python
from algo_stack.supervised.mlp import MLPClassifier, MLPRegressor
```

## Public API

Both classes share the constructor signature:

| Name | Type | Description |
| ---- | ---- | ----------- |
| `MLP*(*, hidden_layer_sizes=(32,), activation="relu", learning_rate=0.01, momentum=0.9, l2=0.0, batch_size=32, n_epochs=200, tol=1e-5, shuffle=True, random_state=None)` | constructor | All keyword-only. `activation` is `"relu"`, `"tanh"` or `"sigmoid"`. |
| `.fit(X, y)`           | method      | Train and return `self`. |
| `.predict(X)`          | method      | Class labels (classifier) or values (regressor). |
| `.predict_proba(X)`    | method      | (classifier only) Softmax probabilities. |
| `.score(X, y)`         | method      | Accuracy (clf) / R² (reg). |
| `.coefs_`              | fitted attr | `list` of weight matrices, one per layer. |
| `.intercepts_`         | fitted attr | `list` of bias vectors, one per layer. |
| `.classes_`            | fitted attr | (classifier) Sorted class labels. |
| `.loss_curve_`         | fitted attr | Per-epoch training loss. |
| `.n_iter_`             | fitted attr | Epochs actually run. |

## Quick start

```python
import numpy as np
from algo_stack.supervised.mlp import MLPClassifier

rng = np.random.default_rng(0)
X = rng.normal(size=(400, 2))
y = (X[:, 0] * X[:, 1] > 0).astype(int)  # XOR-like

clf = MLPClassifier(hidden_layer_sizes=(16, 16), n_epochs=200,
                    learning_rate=0.05, random_state=0).fit(X, y)
print("acc =", clf.score(X, y))
```

## Run the bundled example

```bash
python -m algo_stack.supervised.mlp.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
