# Convolutional Neural Network (toy)

> A minimal 2-D CNN for small grayscale images: Conv → ReLU → MaxPool →
> Flatten → Linear → Softmax, with full back-propagation via im2col.

## Install / import

```python
from algo_stack.supervised.cnn import CNNClassifier
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `CNNClassifier(*, num_filters=4, filter_size=3, pool_size=2, learning_rate=0.05, n_epochs=80, batch_size=16, random_state=None)` | constructor | All keyword-only. |
| `.fit(X, y)` | method | `X` shape `(n_samples, 1, height, width)`. Returns `self`. |
| `.predict(X)` / `.predict_proba(X)` | methods | Standard classifier API. |
| `.score(X, y)` | method | Accuracy. |
| `.conv_W_`, `.conv_b_` | fitted attr | Convolution kernel / bias. |
| `.fc_W_`, `.fc_b_` | fitted attr | Fully-connected head. |
| `.loss_curve_`, `.n_iter_` | fitted attr | Training diagnostics. |

## Quick start

```python
import numpy as np
from algo_stack.supervised.cnn import CNNClassifier

# 8×8 image with a bright horizontal stripe → class 0
img = np.zeros((8, 8))
img[4, :] = 2.0
X = img[None, None, :, :]   # (1, 1, 8, 8)
y = np.array([0])

# Train on a larger synthetic set — see example.py
```

## Run the bundled example

```bash
python -m algo_stack.supervised.cnn.example
```

Trains on horizontal-bar vs. vertical-bar 8×8 patterns.

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
