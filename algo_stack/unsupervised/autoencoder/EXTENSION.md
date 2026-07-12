# Autoencoder — Extension Guide

## 1. Extension surface

- `_encode` / `_decode` / `_forward` / `_backward` — swap architecture here.
- Constructor: `encoding_dim`, `hidden_dim`, `activation`, `optimizer`.

## 2. Common variants

### 2.1 Denoising Autoencoder

Corrupt inputs with noise before encoding; reconstruct the **clean** target.
Add `noise_std` and apply in the training loop only.

### 2.2 Sparse Autoencoder

Add KL sparsity penalty on bottleneck activations to encourage few active
units.

### 2.3 Variational Autoencoder (VAE)

Replace deterministic `z` with `μ, σ`; sample `z ~ N(μ, σ²)`; add KL term
to loss. Requires reparameterisation trick.

### 2.4 Tied weights

Set `decoder_W1_ = encoder_W2_.T` (and optionally share more layers).

## 3. Invariants

- `transform(X).shape == (n_samples, encoding_dim)`.
- `reconstruct(X).shape == X.shape`.

## 4. Pitfalls

- If `encoding_dim >= n_features`, the bottleneck may not compress — MSE can
  still go to ~0 without learning useful structure.
- Scale-sensitive: standardise `X` first for best results.

## 5. Suggested follow-up

- VAE / β-VAE
- Convolutional Autoencoder (for images)
- PCA as a linear baseline (`docs/ROADMAP.md`)
