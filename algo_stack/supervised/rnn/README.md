# Recurrent Neural Network (vanilla RNN, toy)

> Many-to-one sequence classifier using a vanilla RNN cell and
> Back-Propagation Through Time (BPTT).

## Install / import

```python
from algo_stack.supervised.rnn import RNNClassifier
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `RNNClassifier(*, hidden_size=16, learning_rate=0.05, n_epochs=100, batch_size=32, random_state=None)` | constructor | All keyword-only. |
| `.fit(X, y)` | method | `X` shape `(n_samples, seq_len, n_features)`. Returns `self`. |
| `.predict(X)` / `.predict_proba(X)` | methods | Standard classifier API. |
| `.score(X, y)` | method | Accuracy. |
| `.W_xh_`, `.W_hh_`, `.b_h_` | fitted attr | Recurrent weights / bias. |
| `.W_hy_`, `.b_y_` | fitted attr | Output head. |
| `.loss_curve_`, `.n_iter_` | fitted attr | Training diagnostics. |

## Quick start

```python
import numpy as np
from algo_stack.supervised.rnn import RNNClassifier

# 50 sequences of length 10 with 4 features
X = np.random.randn(50, 10, 4)
y = (X[:, -1, 0] > 0).astype(int)

clf = RNNClassifier(hidden_size=16, n_epochs=50).fit(X, y)
print(clf.score(X, y))
```

## Run the bundled example

```bash
python -m algo_stack.supervised.rnn.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
