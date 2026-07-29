# Variational Autoencoder (VAE)

> Probabilistic autoencoder: encoder outputs ``(μ, log σ²)``, latent samples
> via the reparameterisation trick, decoder reconstructs; loss = recon + β·KL.

## Install / import

```python
from algo_stack.unsupervised.vae import VariationalAutoencoder
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `VariationalAutoencoder(*, encoding_dim=4, hidden_dim=16, learning_rate=0.001, n_epochs=200, batch_size=32, beta=1.0, random_state=None)` | constructor | All keyword-only. |
| `.fit(X)` | method | Train on unlabelled `X`. Returns `self`. |
| `.transform(X)` | method | Return latent means `μ`, shape `(n_samples, encoding_dim)`. |
| `.reconstruct(X)` | method | Decode `μ` (no sampling). |
| `.sample(n_samples)` | method | Draw `z ~ N(0,I)` and decode. |
| `.fit_transform(X)` | method | Fit then return `μ`. |
| `.loss_curve_`, `.n_iter_` | fitted attr | Training diagnostics. |

## Quick start

```python
import numpy as np
from algo_stack.unsupervised.vae import VariationalAutoencoder

X = np.random.randn(200, 10)
vae = VariationalAutoencoder(encoding_dim=3, n_epochs=100).fit(X)
codes = vae.transform(X)
samples = vae.sample(5)
```

## Run the bundled example

```bash
python -m algo_stack.unsupervised.vae.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
