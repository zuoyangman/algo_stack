# Mini-Batch K-Means

> Online centroid updates on random mini-batches — faster K-Means for larger `n`.

## Install / import

```python
from algo_stack.unsupervised.minibatch_kmeans import MiniBatchKMeans
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `MiniBatchKMeans(*, n_clusters=8, batch_size=100, n_init=3, max_iter=100, init="k-means++", random_state=None)` | constructor | Keyword-only hyperparameters. |
| `.fit(X)` / `.predict(X)` / `.transform(X)` / `.fit_predict(X)` | method | Same roles as `KMeans`. |
| `.cluster_centers_`, `.labels_`, `.inertia_`, `.n_iter_` | fitted attr | Best restart by inertia. |

## Quick start

See `example.py`.

## Run the bundled example

```bash
python -m algo_stack.unsupervised.minibatch_kmeans.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
