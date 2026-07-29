# Support Vector Machines (SVC / SVR)

> Soft-margin C-Support Vector Classifier trained with Sequential Minimal
> Optimisation (SMO), plus a Kernel-Ridge-style dual regressor.

## Install / import

```python
from algo_stack.supervised.svm import SVC, SVR
```

## Public API

### `SVC`

| Name | Type | Description |
| ---- | ---- | ----------- |
| `SVC(*, C=1.0, kernel="rbf", gamma="scale", max_iter=1000, tol=1e-3, random_state=None)` | constructor | Soft-margin C-SVM. `kernel` ∈ `{"linear","rbf"}`. |
| `.fit(X, y)` | method | Binary SMO; multiclass via One-vs-Rest. |
| `.predict(X)` | method | Class labels. |
| `.decision_function(X)` | method | Signed distance / OvR scores. |
| `.score(X, y)` | method | Accuracy. |
| `.classes_` | fitted | Sorted class labels. |
| `.support_vectors_` | fitted | Rows with nonzero dual coefficient. |
| `.dual_coef_` | fitted | `α_i y_i` on support vectors (binary); object array of OvR duals (multiclass). |
| `.intercept_` | fitted | Bias `b` (scalar or length-`n_classes`). |

### `SVR`

| Name | Type | Description |
| ---- | ---- | ----------- |
| `SVR(*, C=1.0, kernel="rbf", gamma="scale", fit_intercept=True)` | constructor | Kernel Ridge dual with `λ = 1/C`. |
| `.fit(X, y)` / `.predict(X)` / `.score(X, y)` | methods | Closed-form dual solve; R² score. |
| `.support_vectors_` | fitted | All training rows (dense dual). |
| `.dual_coef_` | fitted | Dual coefficients `α`. |
| `.intercept_` | fitted | Bias. |

**SVR design choice.** This module uses the Kernel Ridge closed form
`α = (K + λI)^{-1} y` rather than ε-SVR SMO — clearer and exact for the
squared-loss dual. See [`PRINCIPLE.md`](PRINCIPLE.md).

## Quick start

```python
import numpy as np
from algo_stack.supervised.svm import SVC

X = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]])
y = np.array([0, 0, 1, 1])
print(SVC(kernel="linear", C=10.0, max_iter=200).fit(X, y).predict([[0.5, 0.0]]))
```

## Run the bundled example

```bash
python -m algo_stack.supervised.svm.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
