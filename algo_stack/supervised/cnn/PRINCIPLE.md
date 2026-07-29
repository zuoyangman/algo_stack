# CNN (toy) — Principle

## 1. Problem statement

Convolutional Neural Networks exploit **local connectivity** and **weight
sharing** to learn spatial hierarchies from grid-structured input (images).
This module implements the smallest useful stack for teaching conv + pool +
backprop.

## 2. Mathematical model

**Convolution** (single input channel, one filter):

```
(O * I)(i, j) = Σ_{u,v} I(i+u, j+v) · K(u, v) + b
```

**Max pooling** over `p×p` windows takes the maximum activation.

**Classification head**: flatten conv feature maps → affine map → softmax.

Loss: mean categorical cross-entropy.

## 3. Derivation — back-propagation

### Convolution via im2col

Unfold overlapping patches into columns so convolution becomes a matrix
multiply: `Y = W_col @ cols`. Backward:

```
dW = dY @ colsᵀ,    dcols = W_colᵀ @ dY,    dX = col2im(dcols)
```

### Max-pool backward

Route gradient only to the argmax position in each window (others get 0).

### ReLU backward

`dZ = dA ⊙ (A > 0)`.

### FC + softmax + CE

Same as Softmax Classifier: `δ = (P − Y) / n`.

## 4. Algorithm

```
for epoch:
    for mini-batch:
        X → Conv → ReLU → MaxPool → Flatten → FC → Softmax
        backprop chain → update conv_W, conv_b, fc_W, fc_b
```

## 5. Complexity

| Layer | Forward (per sample) |
| ----- | -------------------- |
| Conv 3×3 on H×W | `O(C_out · C_in · k² · H · W)` |
| MaxPool 2×2 | `O(C · H · W)` |
| FC | `O(flat_dim · K)` |

## 6. Implementation notes

- im2col trades memory for speed and clarity — the standard teaching approach.
- `fc_W_` is lazily initialised on the first forward pass once the flattened
  dimension is known (depends on input image size and architecture).
- Input must be 4-D `(N, C, H, W)` with `C=1` for this toy version.
