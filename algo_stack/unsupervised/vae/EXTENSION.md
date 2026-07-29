# Variational Autoencoder — Extension Guide

## 1. Extension surface

- `_forward` / `_backward` — swap activations, deepen encoder/decoder, or
  change the reconstruction likelihood (BCE for binary data).
- Constructor: `encoding_dim`, `hidden_dim`, `beta`, `learning_rate`.

## 2. Common variants

### 2.1 β-VAE

Increase `beta` to encourage more disentangled latents (at the cost of
reconstruction fidelity).

### 2.2 Conditional VAE

Concatenate labels into encoder and decoder inputs for class-conditional
generation.

### 2.3 Convolutional VAE

Replace dense layers with conv / transpose-conv for images.

### 2.4 Free-bits / annealing

Warm up `beta` from 0 → target over early epochs to avoid posterior collapse.

## 3. Invariants

- `transform(X).shape == (n_samples, encoding_dim)`.
- `reconstruct(X).shape == X.shape`.
- `sample(n).shape == (n, n_features_in_)`.

## 4. Pitfalls

- Posterior collapse: if KL dominates, `μ → 0` and the decoder ignores `z`.
  Lower `beta` or anneal.
- Scale-sensitive: standardise `X` first for best results.

## 5. Suggested follow-up

- Denoising / sparse autoencoders
- Hierarchical VAE
- Adversarial autoencoder
