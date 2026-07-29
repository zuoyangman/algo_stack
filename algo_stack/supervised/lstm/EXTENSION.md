# LSTM / GRU — Extension Guide

## 1. Extension surface

- `_forward` / `_backward` — swap gate parameterisations or add peephole
  connections.
- Shared hyperparameters: `hidden_size`, `learning_rate`, `n_epochs`,
  `batch_size`.

## 2. Common variants

### 2.1 Bidirectional LSTM / GRU

Run forward and backward cells; concatenate final states before the head.

### 2.2 Many-to-many / sequence labelling

Emit a prediction at every time step; average CE over `T`.

### 2.3 Peephole LSTM

Add cell-state connections into the gate pre-activations.

### 2.4 Truncated BPTT

Only back-propagate over the last `τ` steps on long sequences.

## 3. Invariants

- `X` shape `(N, T, d)`; `fit` does not mutate `X`.
- Cache list has length `T`; index `t` corresponds to input `x_t`.
- Probabilities sum to 1 across classes.

## 4. Pitfalls

- Exploding gates on very long `T` — keep gradient clipping enabled.
- Large `hidden_size` with small data overfits quickly; start small.

## 5. Suggested follow-up

- Seq2Seq encoder–decoder with attention
- LayerNorm / residual gated cells
- 1-D CNN front-end + LSTM
