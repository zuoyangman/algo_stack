# Variational Autoencoder — Principle

## 1. Problem statement

A deterministic autoencoder maps `x → z → x̂` but does not define a generative
prior over `z`. A VAE models `q_φ(z|x) = N(μ_φ(x), σ²_φ(x))` and
`p(z) = N(0, I)`, enabling both representation learning and sampling.

## 2. Mathematical model

```
h = relu(x W₁ + b₁)
μ = h W_μ + b_μ,    log σ² = h W_λ + b_λ
z = μ + σ ⊙ ε,      ε ~ N(0, I)          (reparameterisation)
x̂ = decoder(z)
L = ||x − x̂||² + β · KL(q(z|x) || p(z))
```

Closed-form KL:

```
KL = ½ Σⱼ (μⱼ² + σⱼ² − 1 − log σⱼ²)
```

## 3. Derivation

The reconstruction term back-propagates through the decoder into `z`, then
splits into `μ` and `log σ²` via `z = μ + σ ⊙ ε`. The KL term contributes
analytic gradients `∂KL/∂μ = μ` and `∂KL/∂log σ² = ½(σ² − 1)`.

## 4. Algorithm

```
for epoch:
    for mini-batch:
        encode → (μ, log σ²) → sample z → decode → x̂
        loss = MSE(x̂, x) + β · KL
        Adam step on all encoder / decoder parameters
```

## 5. Complexity

Same order as a 4-layer MLP: `O(n · d · h)` per epoch, plus `O(n · e)` for
the KL term (`e = encoding_dim`).

## 6. Notes

- `transform` returns **μ** (deterministic); `sample` draws from the prior.
- `log σ²` is clipped to `[-10, 10]` for numerical stability.
- `beta` implements β-VAE; `beta=1` recovers the standard ELBO weighting
  (up to the choice of reconstruction likelihood / MSE proxy).
