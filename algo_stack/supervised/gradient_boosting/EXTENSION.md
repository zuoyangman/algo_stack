# Gradient Boosting — Extension Guide

## 1. Extension surface

- **Loss**: add Huber, quantile, multinomial deviance; change residual
  computation in the boosting loop.
- **Base learner**: already `DecisionTreeRegressor`; swap for stumps-only
  or histogram trees.
- **Line search / Newton leaves**: after growing tree structure on gradients,
  set leaf values to optimal step for the loss (Friedman).

## 2. Common variants

### 2.1 Newton leaf values (logistic)

For samples in leaf `R`, set

```
γ = Σ_{i∈R} (y_i − p_i) / Σ_{i∈R} p_i (1 − p_i)
```

Improves convergence vs raw residual means.

### 2.2 Stochastic / subsampled boosting

Fit each tree on a random row subsample (and optionally feature subsample).

### 2.3 Multiclass

One-vs-rest or K-tree softmax stages per iteration.

## 3. Testing checklist

- Regressor R² > 0.9 on a mildly noisy sine.
- Binary classifier separates blobs with high accuracy and valid proba.
- `len(estimators_) == n_estimators`.
