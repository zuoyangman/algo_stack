# Convolutional Autoencoder — Principle

## 1. Problem statement

Learn a compressed representation of small images by reconstructing them
through a **spatial encoder–decoder** with at least one convolution, rather
than a fully-connected autoencoder on flattened pixels.

## 2. Architecture

Input `X ∈ ℝ^{N×1×H×W}` (e.g. 8×8 grayscale).

```
Encoder:
  Conv (same-pad, F filters) → ReLU → MaxPool (2×2)
  Flatten → Dense → z ∈ ℝ^{encoding_dim}

Decoder:
  Dense → reshape (F, H/2, W/2)
  Nearest-neighbour upsample ×2
  Conv (same-pad, F) → ReLU
  Conv (same-pad, 1 channel) → x̂
```

## 3. Loss

Mean squared reconstruction error:

```
L = (1 / N) Σᵢ ‖xᵢ − x̂ᵢ‖²_F
```

Gradients flow through both convolutions, pool, upsample, and dense layers
via standard backprop (im2col conv helpers shared with the toy CNN).

## 4. Algorithm

```
init conv / dense weights (He-style)
for epoch:
  for mini-batch:
    encode → decode → MSE gradient
    clip global grad norm; SGD step
```

## 5. Complexity

Per sample roughly `O(F · k² · H · W + flat_dim · encoding_dim)` plus the
symmetric decoder cost. Designed for tiny images (tests use 8×8).

## 6. Notes

- Same-padding keeps spatial size through conv layers; pool halves H/W.
- `transform` returns the bottleneck; `reconstruct` runs the full path.
- Single-channel only in this didactic build (`C=1`).
