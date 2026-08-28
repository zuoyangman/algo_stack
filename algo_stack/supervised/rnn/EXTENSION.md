# RNN (vanilla) — Extension Guide

## 1. Extension surface

- `_forward` / `_backward` — add new cell types (LSTM, GRU) as alternate
  methods or subclasses.
- `hidden_size`, `learning_rate`, training loop hyperparameters.

## 2. Common variants

### 2.1 LSTM / GRU cell

Replace the `tanh` update with gated recurrence; cache gate activations in
`_forward` for backward.

### 2.2 Many-to-many / sequence labelling

Emit a prediction at every time step; loss is averaged over `T`.

### 2.3 Bidirectional RNN

Run forward and backward RNNs; concatenate `h_t_fwd` and `h_t_bwd` before
the output head.

### 2.4 Truncated BPTT

Only back-propagate over the last `τ` steps to save memory on long sequences.

## 3. Invariants

- `X` shape `(N, T, d)`; `fit` does not mutate `X`.
- Hidden states list has length `T`; index `t` corresponds to input `x_t`.

## 4. Pitfalls

- Exploding/vanishing gradients for `T > 50` — clip gradients or use LSTM.
- Shuffled mini-batches across unrelated sequences are fine; state is reset
  per sequence (not carried across batch items).

## 5. Suggested follow-up

- LSTM / GRU
- Seq2Seq with attention
- 1-D CNN for sequences (parallel conv features)
