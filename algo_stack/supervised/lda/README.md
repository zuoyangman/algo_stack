# Linear Discriminant Analysis (LDA)

> Closed-form generative classifier assuming shared Gaussian class covariances.
> Decision boundaries are linear hyperplanes in feature space.

## Install / import

```python
from algo_stack.supervised.lda import LinearDiscriminantAnalysis
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `LinearDiscriminantAnalysis(*, priors=None, shrinkage=None)` | constructor | Optional class priors and covariance shrinkage. |
| `.fit(X, y)` | method | Estimate means, pooled cov, linear coeffs. |
| `.predict(X)` | method | Predicted class labels. |
| `.predict_proba(X)` | method | Softmax over discriminant scores. |
| `.decision_function(X)` | method | Discriminant scores `(n_samples, n_classes)`. |
| `.score(X, y)` | method | Accuracy. |
| `.classes_` / `.means_` / `.cov_` / `.priors_` | fitted | Generative parameters. |
| `.coef_` / `.intercept_` | fitted | Linear decision parameters. |

## Quick start

```python
import numpy as np
from algo_stack.supervised.lda import LinearDiscriminantAnalysis

rng = np.random.default_rng(0)
X0 = rng.normal(loc=(-2, 0), size=(100, 2))
X1 = rng.normal(loc=(+2, 0), size=(100, 2))
X = np.vstack([X0, X1]); y = np.array([0]*100 + [1]*100)

clf = LinearDiscriminantAnalysis().fit(X, y)
print(clf.score(X, y))
```

## Run the bundled example

```bash
python -m algo_stack.supervised.lda.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
