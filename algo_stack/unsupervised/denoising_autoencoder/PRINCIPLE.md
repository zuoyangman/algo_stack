# Denoising Autoencoder — Principle

## 1. Problem statement

A standard autoencoder can memorise identity when capacity is high. A
denoising autoencoder (DAE) is trained to reconstruct the **clean** input
from a **corrupted** observation, forcing the code to capture stable
structure rather than noise.

## 2. Mathematical model

```
X̃ = X + ε,    ε ~ N(0, σ_noise² I)     (training only)
z  = enc(X̃)
X̂  = dec(z)
L  = ||X − X̂||²                         (target is clean X)
```

Architecture matches the symmetric bottleneck autoencoder (two encoder
layers, two decoder layers, ReLU hidden activations).

## 3. Derivation

Same as the MSE autoencoder: `∂L/∂X̂ = 2(X̂ − X) / n`, back-propagated through
decoder then encoder. The only training difference is that the encoder sees
`X̃` while the loss compares against clean `X`.

## 4. Algorithm

```
for epoch:
    for mini-batch Xb:
        X̃ = Xb + N(0, noise_std²)
        forward: encode(X̃) → decode → X̂
        backward: MSE(X̂, Xb) through both halves
        momentum SGD step
```

## 5. Complexity

Identical to a 4-layer autoencoder: `O(n · d · h)` per epoch.

## 6. Notes

- Noise is applied **only** during `fit`; `transform` / `reconstruct` use
  clean inputs.
- `noise_std=0` recovers ordinary autoencoder training.
