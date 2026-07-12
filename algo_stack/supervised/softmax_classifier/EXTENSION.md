# Softmax Classifier — Extension Guide

## 1. Extension surface

- `optimizer` dispatch via `optim.make_optimizer` — add new optimisers there.
- Gradient block in `fit` — single place to add regularisers or loss variants.
- `decision_function` / `predict_proba` — keep signatures stable.

## 2. Common variants

### 2.1 Hinge loss (Linear SVM)

Replace cross-entropy with multi-class hinge; gradient becomes piecewise
linear. Override the `delta = (probs - Yb) / n_b` line.

### 2.2 Temperature scaling

Add `temperature: float = 1.0`; divide logits by `T` before softmax for
calibration experiments.

### 2.3 Feature normalisation pipeline

Compose with `StandardScaler` outside the model, or add `normalize=True` that
fits a scaler internally (not recommended — prefer explicit pipelines).

## 3. Invariants

- `coef_[k]` corresponds to `classes_[k]`.
- Gradients are mean-scaled (divided by batch size) so `learning_rate` is
  batch-size invariant.

## 4. Pitfalls

- Adam needs `reset()` called once before the first `step()` — `fit` handles
  this; if you add warm-start, call `reset` when parameter shapes change.
- Do not penalise `intercept_` with L2.

## 5. Suggested follow-up

- Multi-Layer Perceptron (non-linear features)
- Convolutional Neural Network (`supervised/cnn`)
