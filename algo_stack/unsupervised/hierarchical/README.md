# Agglomerative Clustering

> Bottom-up hierarchical clustering with single / complete / average / Ward linkage.

## Install / import

```python
from algo_stack.unsupervised.hierarchical import AgglomerativeClustering
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `AgglomerativeClustering(*, n_clusters=2, linkage="ward")` | constructor | Linkages: single, complete, average, ward. |
| `.fit(X)` / `.fit_predict(X)` | method | Build dendrogram then cut to `n_clusters`. |
| `.labels_`, `.n_clusters_`, `.children_` | fitted attr | Labels, requested k, merge history. |

## Quick start

See `example.py`.

## Run the bundled example

```bash
python -m algo_stack.unsupervised.hierarchical.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
