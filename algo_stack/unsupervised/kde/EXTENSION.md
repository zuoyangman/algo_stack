# Kernel Density Estimation — Extension Guide

## 1. Extension surface

- `bandwidth`, `kernel` (only gaussian today).
- Distance / kernel helpers.

## 2. Variants

- Epanechnikov / tophat kernels.
- Bandwidth selection (Scott / Silverman rules).

## 3. Invariants

- `score_samples` returns finite floats for finite inputs when `bandwidth > 0`.
- `score` = mean of `score_samples`.

## 4. Pitfalls

- Bandwidth too small → spiky; too large → oversmoothed.
- Curse of dimensionality: KDE degrades quickly as `d` grows.
