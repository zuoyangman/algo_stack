# t-SNE — Extension Guide

## 1. Extension surface

- `perplexity`, `learning_rate`, `early_exaggeration`, `n_iter`.
- Affinity construction / optimiser.

## 2. Variants

- Barnes-Hut / FIt-SNE for large `n`.
- Symmetric SNE (Gaussian low-d kernel).

## 3. Invariants

- `perplexity < n_samples`.
- Embedding is centred each iteration.

## 4. Pitfalls

- Results are random without `random_state`.
- Too high perplexity merges clusters; too low fragments them.
