# Perceptron — Extension Guide

## 1. Extension surface

- Constructor: `learning_rate`, `n_iter`, `tol`, `shuffle`, `fit_intercept`.
- The per-sample update block inside `fit` — override or extract to
  `_update_sample(x, y, ...)`.
- Multiclass strategy: currently OvR; swap to true multiclass Perceptron
  (single weight matrix + argmax) by changing the mistake-handling block.

## 2. Common variants

### 2.1 Pocket algorithm

Keep the best-so-far weight vector (lowest cumulative mistakes) when data are
*not* linearly separable. Add `best_coef_`, `best_intercept_` attributes
updated whenever `mistakes` improves.

### 2.2 Averaged Perceptron

Maintain a running average of weight vectors across all updates; use the
averaged weights for prediction (often better generalisation). Add
`average=True` and track `sum_w`, `count`.

### 2.3 Kernel Perceptron

Replace `x_i` with `φ(x_i)` or evaluate inner products via a kernel
`K(x_i, x_j)`. The update becomes `α_i += η · y_i` in dual form.

### 2.4 Soft-margin / voted Perceptron

See Pocket (§2.1) and Averaged (§2.2).

## 3. Invariants

- `fit` returns `self`; inputs are not mutated.
- `predict` outputs labels from `classes_`, not internal indices.
- Binary `coef_` has shape `(n_features,)`; multiclass `(n_classes, n_features)`.

## 4. Pitfalls

- Do not expect convergence on XOR — use `MLPClassifier` instead.
- Very large `learning_rate` can cause oscillation on separable data near the
  margin boundary.
- OvR multiclass does not have the same convergence guarantee as the binary
  Perceptron.

## 5. Suggested follow-up algorithms

- Multi-Layer Perceptron (`supervised/mlp`)
- Softmax Classifier (`supervised/softmax_classifier`)
- Support Vector Machine (max-margin linear separator)
