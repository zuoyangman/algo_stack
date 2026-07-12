# CNN (toy) — Extension Guide

## 1. Extension surface

- Layer stack in `_forward` / `_backward` — add new layer types as cache
  entries with `type` tags.
- `_im2col` / `_col2im` — extend with padding (`pad > 0`) and `dilation`.
- `num_filters`, `filter_size`, `pool_size` constructor args.

## 2. Common variants

### 2.1 Padding & stride

Add `pad: int` and `stride: int` to convolution; pad `X` with zeros before
`im2col`.

### 2.2 Multiple conv layers

Chain Conv→ReLU→Pool blocks; extend `caches` list and walk it in reverse in
`_backward`.

### 2.3 Batch normalisation

Insert after each conv: normalise per mini-batch, learn γ/β, cache mean/var
for backward.

### 2.4 RGB input

Support `C_in > 1` — the im2col code already handles arbitrary `C`; update
docs and tests.

## 3. Invariants

- `X` is never mutated; 4-D float64 after validation.
- Max-pool backward uses stored argmax indices — must match forward pass
  exactly.

## 4. Pitfalls

- Image size must be large enough after conv+pool: with 3×3 conv and 2×2 pool
  on 8×8, flat dim = `num_filters * 3 * 3`.
- Very small batches make BN unstable.
- im2col memory is `O(C·k²·N·H_out·W_out)` — watch for large images.

## 5. Suggested follow-up

- Deeper CNN (VGG-style blocks)
- 1-D CNN for sequences
- Residual connections (ResNet block)
