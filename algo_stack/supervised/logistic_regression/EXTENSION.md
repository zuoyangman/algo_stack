# Logistic Regression — Extension Guide

## 1. Extension surface

- **Constructor**: add new keyword-only hyperparameters
  (`l1`, `elastic_l1_ratio`, `class_weight`, `optimizer`, …).
- **`fit` loop**: the two branches (binary vs. multinomial) are intentionally
  separate. Sub-classing and overriding `_step(...)` is acceptable; do not
  collapse them into one path without comprehensive tests.
- **`predict_proba` / `decision_function`**: do not change signatures.

## 2. Common variants and how to add them

### 2.1 Mini-batch / stochastic gradient descent

Add `batch_size: int | None = None` (None means full-batch, current
behaviour). Inside the training loop, instead of using the whole `X`, sample
indices with `rng.choice(n, batch_size, replace=False)`.
Critical: still scale the gradient by `1 / batch_size` so the effective
learning rate stays comparable.

### 2.2 L1 / ElasticNet penalty

L1's gradient at 0 is undefined. Use **proximal gradient** (a.k.a. ISTA):
after the gradient step on the smooth part, apply the soft-threshold
operator to `W` row-by-row. Add `l1` and `elastic_l1_ratio` hyperparameters.

### 2.3 Class weights / cost-sensitive learning

Add `class_weight: dict | "balanced" | None = None`. Multiply each sample's
loss / gradient contribution by its class weight. The cleanest place is
right where we form `err = (p - y_one_hot) / n`: instead multiply by
`w_i / Σ w_i` to keep the gradient mean-1.

### 2.4 Newton / IRLS solver

Replace the gradient step by a Newton step `W -= H⁻¹ ∇`, where for binary
LR the Hessian is `Xᵀ diag(p(1-p)) X`. Expose via `solver="newton"`. Mind
that this is `O(n d² + d³)` per iter — only viable for `d` small.

### 2.5 Replacing the optimiser with Adam / Momentum

Refactor: extract the parameter-update line out of `fit` into a small
optimiser class (e.g. `_SGDOptimizer`, `_AdamOptimizer`). Then add an
`optimizer="sgd"|"adam"` switch. This is the same refactor you will want to
do for the MLP, so consider promoting the optimiser to
`algo_stack/utils/optim.py`.

## 3. Invariants the implementation relies on

- `fit` returns `self`, never mutates `X` / `y`.
- `classes_` is sorted (because `np.unique` returns sorted values); the
  i-th row of `coef_` corresponds to `classes_[i]` in the multinomial case.
- `predict` outputs labels drawn from `classes_` *not* the internal integer
  indices.
- `loss_history_` length equals `n_iter_`.

## 4. Pitfalls

- Forgetting to divide gradients by `n` makes the learning rate dataset-size
  dependent.
- Adding L2 to the bias visibly biases the decision boundary towards the
  origin: don't do it.
- Using `np.exp` without the log-sum-exp trick will overflow on any
  reasonably large logit (e.g. > 700 in double precision).

## 5. Suggested follow-up algorithms

- Softmax classifier as its own standalone module
- Perceptron / hinge-loss linear classifier
- Linear Discriminant Analysis (generative counterpart)
- Bayesian Logistic Regression
