# DBSCAN

> Density-based clustering that finds arbitrarily shaped clusters and marks sparse points as noise.

## Install / import

```python
from algo_stack.unsupervised.dbscan import DBSCAN
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `DBSCAN(*, eps=0.5, min_samples=5, metric="euclidean")` | constructor | Keyword-only. |
| `.fit(X)` / `.fit_predict(X)` | method | Fit; labels include `-1` for noise. |
| `.labels_`, `.core_sample_indices_` | fitted attr | Cluster ids and core indices. |

## Quick start

See `example.py`.

## Run the bundled example

```bash
python -m algo_stack.unsupervised.dbscan.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
