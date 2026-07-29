# Gaussian Mixture (EM)

> Soft clustering via a finite mixture of Gaussians trained with EM.

## Install / import

```python
from algo_stack.unsupervised.gmm import GaussianMixture
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `GaussianMixture(*, n_components=1, max_iter=100, tol=1e-3, covariance_type="diag", reg_covar=1e-6, random_state=None)` | constructor | `covariance_type` is `diag` or `full`. |
| `.fit(X)` / `.predict(X)` / `.predict_proba(X)` / `.score_samples(X)` / `.score(X)` | method | EM fit; hard labels; responsibilities; log-likelihood. |
| `.weights_`, `.means_`, `.covariances_`, `.responsibilities_`, `.labels_`, `.n_iter_`, `.lower_bound_` | fitted attr | Mixture parameters and last E-step. |

## Quick start

See `example.py`.

## Run the bundled example

```bash
python -m algo_stack.unsupervised.gmm.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
