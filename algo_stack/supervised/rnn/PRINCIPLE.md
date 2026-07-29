# RNN (vanilla) — Principle

## 1. Problem statement

Sequences `x_1, …, x_T` carry information across time. A vanilla RNN maintains
a hidden state `h_t` that summarises the past and enables many-to-one
classification from the final state `h_T`.

## 2. Mathematical model

```
h_t = tanh(x_t W_xh + h_{t-1} W_hh + b_h),    h_0 = 0
ŷ   = softmax(h_T W_hy + b_y)
```

Loss: mean cross-entropy between `ŷ` and the true class.

## 3. Derivation — BPTT

Unroll the network through time. The output-layer gradient is the same as
softmax+CE: `δ_T = (P − Y) / n`.

For `t = T−1 … 0`, back-propagate through the tanh cell:

```
d_z_t = d_h_t ⊙ tanh'(h_t)
d_W_xh += x_tᵀ d_z_t
d_W_hh += h_{t-1}ᵀ d_z_t
d_h_{t-1} = d_z_t W_hhᵀ
```

This is **Back-Propagation Through Time (BPTT)**.

## 4. Algorithm

```
for epoch:
    for mini-batch of sequences:
        forward: compute h_1..h_T, probs
        backward: BPTT from t=T down to t=0
        SGD update on W_xh, W_hh, b_h, W_hy, b_y
```

## 5. Complexity

| Quantity | Cost |
| -------- | ---- |
| Forward / BPTT per sample | `O(T · (d·h + h² + h·K))` |
| Memory | `O(T · h)` for stored hidden states |

## 6. Implementation notes

- `h_0` is fixed at zeros (no learnable initial state in this toy version).
- Many-to-one only: the label is predicted from the **last** hidden state.
- Vanishing gradients over long `T` are a known limitation → LSTM/GRU are
  the natural extensions.
