# Denoising Autoencoder

> Autoencoder trained to reconstruct clean inputs from Gaussian-corrupted
> observations — learns robust representations.

## Install / import

```python
from algo_stack.unsupervised.denoising_autoencoder import DenoisingAutoencoder
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `DenoisingAutoencoder(*, encoding_dim=4, hidden_dim=16, noise_std=0.1, learning_rate=0.001, n_epochs=200, batch_size=32, random_state=None)` | constructor | All keyword-only. |
| `.fit(X)` | method | Corrupt `X` during training; reconstruct clean targets. |
| `.transform(X)` | method | Return bottleneck codes. |
| `.reconstruct(X)` | method | Reconstruct (no noise at inference). |
| `.fit_transform(X)` | method | Fit then return codes. |
| `.loss_curve_`, `.n_iter_` | fitted attr | Training diagnostics. |

## Quick start

```python
import numpy as np
from algo_stack.unsupervised.denoising_autoencoder import DenoisingAutoencoder

X = np.random.randn(200, 10)
dae = DenoisingAutoencoder(encoding_dim=3, noise_std=0.2, n_epochs=100).fit(X)
codes = dae.transform(X)
recon = dae.reconstruct(X)
```

## Run the bundled example

```bash
python -m algo_stack.unsupervised.denoising_autoencoder.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
