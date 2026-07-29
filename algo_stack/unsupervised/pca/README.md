# PCA

> Linear dimensionality reduction by truncated SVD of centred data.

## Install / import

```python
from algo_stack.unsupervised.pca import PCA
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `PCA(*, n_components=None)` | constructor | `None` keeps `min(n_samples, n_features)`. |
| `.fit(X)` / `.transform(X)` / `.inverse_transform(X)` / `.fit_transform(X)` | method | Centre, project, reconstruct. |
| `.components_`, `.explained_variance_`, `.explained_variance_ratio_`, `.mean_`, `.n_components_` | fitted attr | Principal axes and variance stats. |

## Quick start

See `example.py`.

## Run the bundled example

```bash
python -m algo_stack.unsupervised.pca.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
