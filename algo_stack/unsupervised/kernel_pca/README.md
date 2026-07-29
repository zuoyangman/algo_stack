# Kernel PCA

> Non-linear PCA via eigendecomposition of a centred Gram matrix.

## Install / import

```python
from algo_stack.unsupervised.kernel_pca import KernelPCA
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `KernelPCA(*, n_components=None, kernel="rbf", gamma=None, degree=3)` | constructor | Kernels: rbf, linear, poly. |
| `.fit(X)` / `.transform(X)` / `.fit_transform(X)` | method | Fit on training Gram; project new points with centred `K(X, X_fit)`. |
| `.alphas_`, `.lambdas_`, `.X_fit_`, `.n_components_` | fitted attr | Scaled eigenvectors / eigenvalues. |

## Quick start

See `example.py`.

## Run the bundled example

```bash
python -m algo_stack.unsupervised.kernel_pca.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
