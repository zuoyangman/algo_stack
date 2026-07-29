# Denoising Autoencoder — Extension Guide

## 1. Extension surface

- `_forward` / `_backward` — change depth, activations, or corruption process.
- Constructor: `encoding_dim`, `hidden_dim`, `noise_std`, `learning_rate`.

## 2. Common variants

### 2.1 Masking noise

Zero out a random fraction of features instead of additive Gaussian noise.

### 2.2 Salt-and-pepper

Flip features to domain extremes with small probability (binary / image data).

### 2.3 Stacked DAE

Train layer-wise denoising autoencoders and stack the encoders.

### 2.4 Contractive Autoencoder

Add a Frobenius penalty on the Jacobian of the encoder instead of (or with)
input noise.

## 3. Invariants

- `transform(X).shape == (n_samples, encoding_dim)`.
- `reconstruct(X).shape == X.shape`.
- Training corrupts inputs; inference does not.

## 4. Pitfalls

- Too large `noise_std` can destroy signal so reconstruction never improves.
- Scale-sensitive: standardise `X` so `noise_std` is meaningful relative to
  feature scale.

## 5. Suggested follow-up

- VAE / β-VAE
- Sparse autoencoder
- Convolutional DAE for images
