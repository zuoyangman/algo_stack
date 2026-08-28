# Logistic Regression — Principle

## 1. Problem statement

We want to model `P(y = c | x)` as a linear function of `x` followed by a
suitable activation. Given training data `{(x_i, y_i)}` with `y_i ∈ {0,…,K-1}`
we maximise the conditional log-likelihood (equivalently, minimise the mean
cross-entropy loss).

## 2. Mathematical model

### Binary case (`K = 2`)

```
P(y = 1 | x) = σ(wᵀx + b),     σ(z) = 1 / (1 + e^{-z}).
```

Loss (mean binary cross-entropy):

```
L(w, b) = -1/n · Σ_i [ y_i log σ(zᵢ) + (1 - y_i) log(1 - σ(zᵢ)) ],   zᵢ = wᵀxᵢ + b.
```

### Multinomial case (`K ≥ 3`)

```
P(y = k | x) = softmax(W x + b)_k = exp(w_kᵀx + b_k) / Σ_j exp(w_jᵀx + b_j).
```

Loss (mean categorical cross-entropy with one-hot target `Y`):

```
L(W, b) = -1/n · Σ_i Σ_k Y_{i,k} log P(y = k | x_i).
```

We optionally add an L2 penalty `(α / (2n)) · ||W||_F²` to the weights (the
bias is **not** penalised). The mean-over-`n` scaling makes the gradient and
loss invariant to the batch size.

## 3. Derivation

For the binary case, the gradient of `L` simplifies beautifully:

```
∂L/∂w = (1/n) Σ_i (σ(zᵢ) - y_i) xᵢ = (1/n) Xᵀ (p - y),
∂L/∂b = (1/n) Σ_i (σ(zᵢ) - y_i).
```

Adding L2: `∂L/∂w += (α/n) w`. Same form for the multinomial case with
`p = softmax(...)` and one-hot `y`.

## 4. Algorithm

```
1. Validate inputs and one-hot encode y if K > 2.
2. Initialise W, b to zeros.
3. For t = 1 .. n_iter:
     z = X @ Wᵀ + b
     p = softmax(z)   # or sigmoid(z) in the binary path
     compute mean cross-entropy loss (+ L2 term) → loss_history
     grad_W = (p - Y_onehot).T @ X / n   (+ α/n * W)
     grad_b = (p - Y_onehot).sum(0) / n
     if ||grad||_∞ < tol: break
     W -= lr * grad_W; b -= lr * grad_b
4. Store coef_, intercept_, classes_, n_iter_, loss_history_.
```

## 5. Complexity

| Quantity | Cost |
| -------- | ---- |
| Training time   | `O(n_iter · n · d · K)`  |
| Training memory | `O(n · K)` for the probability matrix. |
| Prediction time | `O(d · K)` per sample. |

## 6. Numerical & implementation notes

- Sigmoid is implemented branchwise (`z ≥ 0` vs. `z < 0`) to avoid overflow
  in `exp(-z)` for large negative `z`.
- Softmax subtracts the per-row max before exponentiating (log-sum-exp trick).
- Loss adds a tiny `eps = 1e-12` inside the `log` to avoid `-inf` when a
  predicted probability collapses to 0; a more rigorous version uses
  `logsumexp`-based loss formulations.
- We start from zero weights. Because the loss is convex this is a perfectly
  fine initialisation and makes results deterministic.
- The gradient is averaged over `n` so `learning_rate` semantics are
  independent of batch size.
