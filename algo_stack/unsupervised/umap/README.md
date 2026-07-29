# UMAP

> Simplified didactic Uniform Manifold Approximation and Projection for small
> datasets (dense fuzzy knn graph + SGD layout).

## Install / import

```python
from algo_stack.unsupervised.umap import UMAP
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `UMAP(*, n_components=2, n_neighbors=15, min_dist=0.1, n_epochs=200, learning_rate=1.0, random_state=None, init="spectral")` | constructor | Keyword-only. |
| `.fit_transform(X)` / `.fit(X)` | method | Build fuzzy graph and optimise embedding. |
| `.transform(X)` | method | Returns `.embedding_` when `n_samples` matches training; OOS raises. |
| `.embedding_`, `.graph_`, `.a_`, `.b_` | fitted attr | Low-d map, fuzzy memberships, curve params. |

## Quick start

```python
import numpy as np
from algo_stack.unsupervised.umap import UMAP

X = np.random.randn(80, 4)
Y = UMAP(n_neighbors=10, n_epochs=100, random_state=0).fit_transform(X)
```

## Run the bundled example

```bash
python -m algo_stack.unsupervised.umap.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
