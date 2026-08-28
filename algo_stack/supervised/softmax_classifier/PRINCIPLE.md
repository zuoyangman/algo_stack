# Softmax Classifier — Principle

## 1. Problem statement

Given labelled samples `(x_i, y_i)` with `y_i ∈ {0,…,K−1}`, learn a linear
scoring function `s_k(x) = w_k·x + b_k` for each class `k`, then convert
scores to probabilities via softmax:

```
P(y = k | x) = exp(s_k) / Σ_j exp(s_j).
```

Training minimises the mean cross-entropy loss.

## 2. Mathematical model

```
L(W, b) = −1/n · Σ_i log P(y = y_i | x_i)  +  (λ/2n) · ||W||_F²
```

where `W` stacks all class weight vectors and `λ` is the L2 coefficient.

## 3. Derivation

For one sample with one-hot target `e_k`:

```
∂L/∂s_j = P(y=j|x) − 1[j=k]
```

Batch gradient:

```
∂L/∂W = (P − Y)ᵀ X / n,    ∂L/∂b = sum(P − Y) / n
```

This is identical to multinomial logistic regression; the distinction here
is **engineering** (mini-batch SGD, pluggable optimisers from
`algo_stack.utils.optim`) rather than statistical framing.

## 4. Algorithm

```
Initialise W, b with small random values
for epoch in 1..n_epochs:
    for each mini-batch (X_b, Y_b):
        P = softmax(X_b Wᵀ + b)
        δ = (P − Y_b) / |batch|
        W -= opt_step(δᵀ X_b + λ W)
        b   -= opt_step(sum δ)
    record full-data cross-entropy → loss_curve
```

## 5. Complexity

| Quantity | Cost |
| -------- | ---- |
| Per mini-batch | `O(B · d · K)` |
| Memory | `O(d · K)` parameters |

## 6. Numerical notes

- Softmax uses the log-sum-exp trick via `activations.softmax`.
- Optimisers live in `algo_stack.utils.optim` and are shared with future
  modules (CNN, RNN, Autoencoder).
- Small random init (not zeros) breaks symmetry between classes at start.
