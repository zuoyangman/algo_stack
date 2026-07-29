# Kernel PCA — Extension Guide

## 1. Extension surface

- `kernel`, `gamma`, `degree`.
- `_kernel_matrix` for new kernels.

## 2. Variants

- Sparse KPCA / Nyström approximation.
- Precomputed kernel matrix input.

## 3. Invariants

- Only positive eigenvalues kept.
- `fit_transform` uses the training centred Gram.

## 4. Pitfalls

- `gamma` too large → near-diagonal Gram; too small → overly smooth.
- Memory `O(n²)` limits dataset size.
