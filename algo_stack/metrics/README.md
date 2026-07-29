# Metrics

> Curated evaluation functions for classification, regression, and clustering.

This package is **function-oriented** (not estimator folders). It re-exports
the core scores used by default ``score`` methods and adds common extras.

## Install / import

```python
from algo_stack.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
    precision_recall_f1,
    confusion_matrix,
    silhouette_score,
    log_loss,
)
```

## Catalogue

| Function | Task | Notes |
| -------- | ---- | ----- |
| `accuracy_score(y_true, y_pred)` | classification | Fraction of exact matches. Re-exported from `utils.metrics`. |
| `mean_squared_error(y_true, y_pred)` | regression | Mean `(y − ŷ)²`. Re-exported. |
| `r2_score(y_true, y_pred)` | regression | Coefficient of determination. Re-exported. |
| `precision_recall_f1(y_true, y_pred, *, pos_label=1)` | binary classification | Returns `(precision, recall, f1)`. |
| `confusion_matrix(y_true, y_pred, *, labels=None)` | classification | Rows = true, columns = predicted. |
| `silhouette_score(X, labels)` | clustering | Mean silhouette (Euclidean). Needs ≥ 2 clusters. |
| `log_loss(y_true, y_prob, *, eps=1e-15)` | probabilistic classification | Binary or multiclass cross-entropy. |

## Quick examples

```python
import numpy as np
from algo_stack.metrics import precision_recall_f1, confusion_matrix, log_loss

y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0, 1, 0, 0, 1])
p, r, f1 = precision_recall_f1(y_true, y_pred)
print(p, r, f1)
print(confusion_matrix(y_true, y_pred))

# Binary log loss from probabilities of the positive class
y_prob = np.array([0.1, 0.9, 0.4, 0.2, 0.8])
print(log_loss(y_true, y_prob))
```

```python
import numpy as np
from algo_stack.metrics import silhouette_score

X = np.vstack([
    np.random.randn(20, 2) + np.array([0.0, 0.0]),
    np.random.randn(20, 2) + np.array([5.0, 5.0]),
])
labels = np.array([0] * 20 + [1] * 20)
print(silhouette_score(X, labels))
```

## Design notes

- NumPy only; no scikit-learn dependency.
- Re-exports keep a single import path while leaving
  `algo_stack.utils.metrics` available for estimators' default ``score``.
- See `example.py` for a runnable smoke demo.
