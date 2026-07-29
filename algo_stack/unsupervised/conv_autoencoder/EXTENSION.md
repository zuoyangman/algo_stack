# Convolutional Autoencoder — Extension Guide

## 1. Extension surface

- Constructor: `encoding_dim`, `num_filters`, `filter_size`, `pool_size`,
  `learning_rate`, `n_epochs`, `batch_size`.
- `_encode_forward` / `_decode_forward` / `_backward` — deepen the stack.
- Swap SGD for momentum/Adam via `algo_stack.utils.optim`.

## 2. Variants

- **Tied weights** — share encoder/decoder conv kernels (transposed).
- **Strided conv** instead of max-pool for learned downsampling.
- **Multi-channel** — lift the `C=1` check and init `enc_conv_W_` with `C_in`.
- **Denoising** — corrupt inputs during training (see DenoisingAutoencoder).

## 3. Invariants

- `H` and `W` divisible by `pool_size`.
- `filter_size` odd (same-padding).
- `transform(X).shape[1] == encoding_dim`.
- `reconstruct(X).shape == X.shape`.

## 4. Pitfalls

- Too large `learning_rate` diverges (grad clip at 5 helps but is not enough).
- Very small `encoding_dim` may underfit textured images.
- Upsample is nearest-neighbour (blocky); bilinear would be smoother.

## 5. Suggested follow-up

- Deeper U-Net-style skip connections
- VAE with convolutional encoder/decoder
- Compare against `algo_stack.unsupervised.autoencoder` on flattened images
