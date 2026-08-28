# MLP — Extension Guide

## 1. Extension surface

- **Constructor**: `hidden_layer_sizes`, `activation`, `learning_rate`,
  `momentum`, `l2`, `batch_size`, `n_epochs`, `tol`, `shuffle`,
  `random_state`. New hyperparameters keyword-only, stored unchanged.
- **`_ACT_FORWARD` dict + `_act_grad`**: add a new activation by adding
  both the forward function and the gradient. The gradient must accept the
  *output* `a` (not the pre-activation `z`).
- **`_forward` / `_backward`**: hooks for changing the layer semantics
  (e.g. dropout). Both should remain symmetric: every term used in
  `_backward` must be cached in `_forward`.
- **`_update`**: replace the SGD-with-momentum update to plug in Adam /
  RMSProp / Nesterov. Subclasses should override this rather than
  `_fit_loop`.
- **`_output_activation_` / `_n_outputs_`**: subclasses set these to choose
  the output head + loss. See `MLPRegressor` / `MLPClassifier`.

## 2. Common variants and how to add them

### 2.1 Adam optimiser

Add `optimizer="adam"` to `__init__`; in `_update`, dispatch on
`self.optimizer`. Keep per-parameter `m_W`, `v_W`, `m_b`, `v_b` buffers
created in `_init_params`. Maintain a step counter for bias correction.
Better: extract the optimiser into `algo_stack/utils/optim.py` so it can
be re-used by Logistic Regression.

### 2.2 Dropout

In `_forward`, when training, draw a Bernoulli mask of shape
`a_l.shape` with `p_keep = 1 - dropout_rate`, multiply, and divide by
`p_keep` (inverted dropout). Cache the mask. In `_backward`, multiply the
incoming `δ_l` by the *same* mask before propagating. Disable at inference
time (the simplest signal is a new flag set in `predict`).

### 2.3 Batch normalisation

A new operation between `z_l` and `a_l`. Add running mean/var attributes
analogous to `coefs_`. Backprop through the batch-norm transform requires
caching `(x_hat, std, mean)` per layer; see the
[Ioffe & Szegedy paper](https://arxiv.org/abs/1502.03167) for the gradient.

### 2.4 New (output, loss) pair

Override `_backward`'s base case `delta = (a_L - y) / n` to the correct
formula for your output activation + loss combination. Also override
`_epoch_loss`. The cleanest pattern is a new subclass that sets
`_output_activation_` and provides `_loss_grad`.

### 2.5 Early stopping with validation set

Pass `validation_fraction: float` to `__init__`; in `_fit_loop`, split a
held-out chunk before training and stop early when validation loss
increases for `n_no_improve` consecutive epochs.

## 3. Invariants the implementation relies on

- `fit` returns `self` and does not mutate `X` or `y`.
- `coefs_[l].shape == (layer_sizes[l], layer_sizes[l+1])` for all `l`.
- `_forward(X)` returns `len(coefs_) + 1` activations.
- The base case `delta = (a_L - y) / n` is only correct for the two
  (output, loss) pairs documented in `PRINCIPLE.md` §3.
- Mini-batches inherit dtype from `X` (which is float64 after validation),
  so all weight buffers are float64.

## 4. Pitfalls

- **Re-initialising on every `fit`.** `fit` calls `_init_params`, which
  resets `coefs_`, `intercepts_`, *and* the momentum buffers. There is
  currently **no warm-start**. If you add one (`warm_start=True`), be sure
  to only re-init when shapes change.
- **L2 in the bias** — leave it out unless you know what you're doing.
- **Step size + batch size coupling.** Because gradients are mean-over-`n`
  (not sum), `learning_rate` is roughly batch-size-invariant. Doubling
  batch size still benefits slightly from a larger `learning_rate` though.
- **Vanishing gradients** with sigmoid/tanh on deep networks — switch to
  ReLU + He init, or add batch-norm.

## 5. Suggested follow-up algorithms

- Convolutional Neural Network (Conv2D, MaxPool) — same backprop, different
  layer ops.
- Recurrent Neural Network (BPTT)
- Autoencoder (re-uses MLP, just with `y = X` and a bottleneck layer)
- Layer-class refactor: instead of `coefs_` / `intercepts_` lists, model
  each layer as a class with `forward(x)` / `backward(grad_out)` methods.
  This is the natural next refactor before adding Conv / BatchNorm / Dropout.
