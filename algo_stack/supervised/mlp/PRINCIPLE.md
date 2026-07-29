# MLP — Principle

## 1. Problem statement

Approximate a function `f: ℝᵈ → ℝᵐ` by composing linear maps with element-wise
non-linearities. With enough hidden units, an MLP with one hidden layer is a
universal approximator (Hornik 1989); deeper networks can represent the same
functions with fewer parameters.

## 2. Mathematical model

A network with `L` weight layers maps an input `x ∈ ℝᵈ` as:

```
a_0 = x
for l = 1 .. L:
    z_l = a_{l-1} · W_l + b_l
    a_l = φ_l(z_l)         # element-wise non-linearity
ŷ  = a_L
```

Hidden layers use the same activation `φ` (relu/tanh/sigmoid in this impl).
The output layer uses:

- **Regression**: `φ_L = identity`, loss `L = mean_i ||ŷ_i − y_i||²`.
- **Classification**: `φ_L = softmax`, loss `L = −mean_i Σ_k y_{i,k} log ŷ_{i,k}`
  (cross-entropy with one-hot `y`).

Optional L2 weight decay: `L ← L + (λ / (2 n)) · Σ_l ||W_l||_F²`.

## 3. Derivation — back-propagation

Define `δ_l = ∂L / ∂z_l`. Two key identities:

```
∂L/∂W_l = a_{l-1}ᵀ · δ_l
∂L/∂b_l = sum over batch of δ_l
δ_{l-1} = (δ_l · W_lᵀ) ⊙ φ'_{l-1}(z_{l-1})
```

For both common (output activation, loss) pairs we get the same elegant base case:

- (`identity`, MSE):  `δ_L = (a_L − y) / n`
- (`softmax`, cross-entropy):  `δ_L = (a_L − y_onehot) / n`

This is why the code's `_backward` method has a single base case.

Activation derivatives evaluated on the *output* `a`:

```
relu'(z)    = (a > 0).astype(...)
tanh'(z)    = 1 − a²
sigmoid'(z) = a · (1 − a)
```

## 4. Algorithm — mini-batch SGD with momentum

```
init weights with He (relu) or Xavier (tanh/sigmoid)
init velocity buffers v_W, v_b = 0

for epoch in 1 .. n_epochs:
    optionally shuffle indices
    for each minibatch:
        forward pass → activations
        backward pass → grads_W, grads_b  (add L2 to grads_W)
        v_W ← μ · v_W − η · grads_W
        v_b ← μ · v_b − η · grads_b
        W   += v_W;  b += v_b
    compute full-data loss
    early-stop if |Δloss| < tol
```

## 5. Complexity

For one mini-batch of size `B` and layer sizes `[d, h_1, …, h_{L-1}, m]`:

| Quantity | Cost |
| -------- | ---- |
| Forward / Backward per batch | `O(B · Σ_l h_{l-1} · h_l)` |
| Memory                        | `O(B · max_l h_l)` for activations + `O(Σ_l h_{l-1} · h_l)` for parameters. |

## 6. Numerical & implementation notes

- **He vs. Xavier init.** Picking the right scale is by far the easiest way
  to get an untrained network into a learnable regime. We default to He
  for ReLU and Xavier otherwise.
- **Softmax** is computed with the log-sum-exp trick (subtract per-row max).
- **Sigmoid** clips `z` to `[-500, 500]` before exponentiating; on float64
  this is well below overflow but keeps NaNs out for pathological inputs.
- **Activation gradients** are written as functions of the *output* `a`,
  not the pre-activation `z`. This means the forward pass only has to
  remember `a_l`, halving the storage compared to keeping both `(z_l, a_l)`.
- **Joint output+loss simplification.** The base case `δ_L = (a_L − y)/n`
  is correct for both (identity, MSE) and (softmax, cross-entropy), but
  *not* for (softmax, MSE) or (sigmoid, MSE). If you add a new
  loss/output pair, override the base case in `_backward`.
- The L2 term is added *to the gradient of W only* (biases are not
  penalised) and scaled by `λ / n`, keeping the effective learning rate
  invariant to batch size.
