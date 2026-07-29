# Mini-Batch K-Means — Extension Guide

## 1. Extension surface

- Constructor: `batch_size`, `n_init`, `max_iter`, `init`.
- `_single_run` for alternative online update rules.

## 2. Variants

- Replace uniform sampling with reservoir / importance sampling.
- Skip the final full-data mean polish for pure streaming mode.

## 3. Invariants

- `fit` returns `self`; empty clusters are not left empty after fit.
- Same `random_state` → same centres / inertia.

## 4. Pitfalls

- Too small `batch_size` increases variance of centroid paths.
- Compare inertia to full K-Means only after enough iterations.
