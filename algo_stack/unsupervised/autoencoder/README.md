# Autoencoder

> Unsupervised neural network that compresses inputs through a bottleneck and
> reconstructs them — learns a low-dimensional representation via MSE loss.

## Install / import

```python
from algo_stack.unsupervised.autoencoder import Autoencoder
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `Autoencoder(*, encoding_dim=4, hidden_dim=16, activation="relu", learning_rate=0.01, optimizer="momentum", n_epochs=200, batch_size=32, random_state=None)` | constructor | All keyword-only. |
| `.fit(X)` | method | Train on unlabelled `X`. Returns `self`. |
| `.transform(X)` | method | Return bottleneck codes, shape `(n_samples, encoding_dim)`. |
| `.reconstruct(X)` | method | Return reconstructed inputs. |
| `.fit_transform(X)` | method | Fit then return codes. |
| `encoder_*`, `decoder_*` | fitted attr | Eight weight/bias arrays (two layers each). |
| `.loss_curve_`, `.n_iter_` | fitted attr | Training diagnostics. |

## Quick start

```python
import numpy as np
from algo_stack.unsupervised.autoencoder import Autoencoder

X = np.random.randn(200, 10)
ae = Autoencoder(encoding_dim=3, n_epochs=100).fit(X)
codes = ae.transform(X)
recon = ae.reconstruct(X)
```

## Run the bundled example

```bash
python -m algo_stack.unsupervised.autoencoder.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
