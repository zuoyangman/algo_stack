# K-Means Clustering

> Partition `n` samples into `k` clusters by alternating between assigning
> each point to its nearest centroid and re-computing each centroid as the
> mean of its assigned points (Lloyd's algorithm).

## Install / import

```python
from algo_stack.unsupervised.kmeans import KMeans
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `KMeans(*, n_clusters=8, init="k-means++", n_init=10, max_iter=300, tol=1e-4, random_state=None)` | constructor | All hyperparameters keyword-only. `init` is `"k-means++"` or `"random"`. |
| `.fit(X)`               | method      | Train. Returns `self`. |
| `.predict(X)`           | method      | Assign each row to its nearest centroid. |
| `.transform(X)`         | method      | Return the `(n_samples, n_clusters)` distance matrix to all centroids. |
| `.fit_predict(X)`       | method      | Convenience: fit then return `labels_`. |
| `.cluster_centers_`     | fitted attr | Shape `(n_clusters, n_features)`. |
| `.labels_`              | fitted attr | Cluster assignment for the training data. |
| `.inertia_`             | fitted attr | Sum of squared distances from each point to its centroid (best run). |
| `.n_iter_`              | fitted attr | Iterations taken by the best run. |

## Quick start

```python
import numpy as np
from algo_stack.unsupervised.kmeans import KMeans

rng = np.random.default_rng(0)
centres = np.array([[-4., -4.], [0., 4.], [4., -4.]])
X = np.vstack([c + rng.normal(scale=0.5, size=(100, 2)) for c in centres])

km = KMeans(n_clusters=3, random_state=0).fit(X)
print(km.cluster_centers_)
print(km.inertia_)
```

## Run the bundled example

```bash
python -m algo_stack.unsupervised.kmeans.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
