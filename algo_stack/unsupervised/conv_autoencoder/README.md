# Convolutional Autoencoder

> Unsupervised convolutional autoencoder for small grayscale images
> ``(n_samples, 1, H, W)`` — MSE reconstruction with a dense bottleneck.

## Install / import

```python
from algo_stack.unsupervised.conv_autoencoder import ConvAutoencoder
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `ConvAutoencoder(*, encoding_dim=8, num_filters=4, filter_size=3, pool_size=2, learning_rate=0.05, n_epochs=80, batch_size=16, random_state=None)` | constructor | Keyword-only. |
| `.fit(X)` | method | Train on image batch `X`. Returns `self`. |
| `.transform(X)` | method | Bottleneck codes `(n_samples, encoding_dim)`. |
| `.reconstruct(X)` | method | Reconstructed images, same shape as `X`. |
| `.fit_transform(X)` | method | Fit then return codes. |
| `enc_*`, `dec_*`, `.loss_curve_`, `.n_iter_`, `.image_shape_` | fitted attr | Weights and diagnostics. |

## Quick start

```python
import numpy as np
from algo_stack.unsupervised.conv_autoencoder import ConvAutoencoder

X = np.random.randn(40, 1, 8, 8)
cae = ConvAutoencoder(encoding_dim=4, n_epochs=40, random_state=0).fit(X)
Z = cae.transform(X)
X_hat = cae.reconstruct(X)
```

## Run the bundled example

```bash
python -m algo_stack.unsupervised.conv_autoencoder.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
