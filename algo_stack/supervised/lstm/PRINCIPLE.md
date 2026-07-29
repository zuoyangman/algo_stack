# LSTM / GRU — Principle

## 1. Problem statement

Vanilla RNNs suffer from vanishing gradients on long sequences. Gated cells
(LSTM, GRU) introduce multiplicative gates that control information flow
through time, enabling stable many-to-one sequence classification from the
final hidden state `h_T`.

## 2. Mathematical model — LSTM

Gates (concatenated form):

```
[i, f, o, g]_pre = x_t W_x + h_{t-1} W_h + b
i = σ(i_pre),  f = σ(f_pre),  o = σ(o_pre),  g = tanh(g_pre)
c_t = f ⊙ c_{t-1} + i ⊙ g
h_t = o ⊙ tanh(c_t)
ŷ   = softmax(h_T W_hy + b_y)
```

## 3. Mathematical model — GRU

```
z = σ(x_t W_xz + h_{t-1} W_hz + b_z)          # update
r = σ(x_t W_xr + h_{t-1} W_hr + b_r)          # reset
n = tanh(x_t W_xn + (r ⊙ h_{t-1}) W_hn + b_n) # candidate
h_t = (1 − z) ⊙ n + z ⊙ h_{t-1}
ŷ   = softmax(h_T W_hy + b_y)
```

Loss: mean cross-entropy.

## 4. Derivation — BPTT

Unroll through time. Output gradient is the softmax+CE residual
`δ = (P − Y) / n`. Propagate `d_h` and (for LSTM) `d_c` from `t = T` down to
`t = 1`, applying the local gate Jacobians at each step.

## 5. Algorithm

```
for epoch:
    for mini-batch of sequences:
        forward: compute gates, c_t / h_t (or GRU h_t), probs
        backward: BPTT from t=T down to t=0
        SGD update (with optional gradient clipping)
```

## 6. Complexity

| Quantity | Cost |
| -------- | ---- |
| LSTM forward/BPTT per sample | `O(T · (d·4h + h·4h + h·K))` |
| GRU forward/BPTT per sample | `O(T · (d·3h + h·3h + h·K))` |
| Memory | `O(T · h)` cached activations |

## 7. Implementation notes

- Forget-gate bias of the LSTM is initialised to `1` to encourage retention.
- Gradients are clipped to L2 norm `5` for training stability.
- Many-to-one only: the label is predicted from the **last** hidden state.
