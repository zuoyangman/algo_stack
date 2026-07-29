# t-SNE

> Exact t-SNE embedding for small datasets (dense affinities, no Barnes-Hut).

## Install / import

```python
from algo_stack.unsupervised.tsne import TSNE
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `TSNE(*, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000, early_exaggeration=12.0, random_state=None)` | constructor | Keyword-only. |
| `.fit_transform(X)` / `.fit(X)` | method | Optimise 2-D (or `n_components`) embedding. |
| `.embedding_`, `.kl_divergence_`, `.n_iter_` | fitted attr | Low-d map and final KL. |

## Quick start

See `example.py`.

## Run the bundled example

```bash
python -m algo_stack.unsupervised.tsne.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
