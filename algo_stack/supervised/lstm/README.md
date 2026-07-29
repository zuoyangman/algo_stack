# LSTM / GRU (toy sequence classifiers)

> Many-to-one sequence classifiers using an LSTM or GRU cell and
> Back-Propagation Through Time (BPTT).

## Install / import

```python
from algo_stack.supervised.lstm import LSTMClassifier, GRUClassifier
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `LSTMClassifier(*, hidden_size=16, learning_rate=0.05, n_epochs=100, batch_size=32, random_state=None)` | constructor | All keyword-only. |
| `GRUClassifier(*, …)` | constructor | Same hyperparameters as LSTM. |
| `.fit(X, y)` | method | `X` shape `(n_samples, seq_len, n_features)`. Returns `self`. |
| `.predict(X)` / `.predict_proba(X)` | methods | Standard classifier API. |
| `.score(X, y)` | method | Accuracy. |
| `.loss_curve_`, `.n_iter_` | fitted attr | Training diagnostics. |

## Quick start

```python
import numpy as np
from algo_stack.supervised.lstm import LSTMClassifier

X = np.random.randn(50, 10, 4)
y = (X[:, -1, 0] > 0).astype(int)

clf = LSTMClassifier(hidden_size=16, n_epochs=50).fit(X, y)
print(clf.score(X, y))
```

## Run the bundled example

```bash
python -m algo_stack.supervised.lstm.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
