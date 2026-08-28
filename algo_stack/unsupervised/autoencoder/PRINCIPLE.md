# Autoencoder — Principle

## 1. Problem statement

Learn a compressed representation `z = enc(x)` and a decoder `x̂ = dec(z)`
such that `x̂ ≈ x`, without using labels. The bottleneck forces the network
to capture the most salient structure in the data.

## 2. Mathematical model

```
enc:  x → h₁ → z        (h₁ = σ(x W₁ + b₁),  z = h₁ W₂ + b₂)
dec:  z → h₂ → x̂        (symmetric architecture)
L = (1/n) Σᵢ ||xᵢ − x̂ᵢ||²
```

## 3. Derivation

MSE loss gradient at the output: `∂L/∂x̂ = 2(x̂ − x) / n`. Back-propagate
through the decoder, then through the encoder (tied or untied weights).

## 4. Algorithm

```
for epoch:
    for mini-batch:
        forward: encode → decode → reconstruction
        backward: MSE gradient through both halves
        optimizer step on all 8 parameter arrays
```

## 5. Complexity

Same order as a 4-layer MLP with matching widths: `O(n · d · h)` per epoch.

## 6. Notes

- We use **untied** encoder/decoder weights (simpler to implement and extend).
- `transform` returns the bottleneck `z`; `reconstruct` runs the full
  encode→decode path.
