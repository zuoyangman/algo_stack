# Gaussian Naive Bayes

> Generative classifier assuming class-conditional feature independence and
> univariate Gaussian margins. Fast, closed-form, works well on continuous data.

## Install / import

```python
from algo_stack.supervised.gaussian_nb import GaussianNB
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `GaussianNB(*, var_smoothing=1e-9, priors=None)` | constructor | Variance floor + optional priors. |
| `.fit(X, y)` | method | Estimate means, variances, priors. |
| `.predict(X)` | method | MAP class labels. |
| `.predict_proba(X)` | method | Posterior class probabilities. |
| `.score(X, y)` | method | Accuracy. |
| `.classes_` | fitted | Unique class labels. |
| `.class_prior_` | fitted | Class priors, shape `(n_classes,)`. |
| `.theta_` | fitted | Means, shape `(n_classes, n_features)`. |
| `.var_` | fitted | Variances, shape `(n_classes, n_features)`. |

## Quick start

```python
import numpy as np
from algo_stack.supervised.gaussian_nb import GaussianNB

rng = np.random.default_rng(0)
X0 = rng.normal(loc=-2, size=(80, 2))
X1 = rng.normal(loc=+2, size=(80, 2))
X = np.vstack([X0, X1]); y = np.array([0]*80 + [1]*80)

clf = GaussianNB().fit(X, y)
print(clf.score(X, y), clf.theta_)
```

## Run the bundled example

```bash
python -m algo_stack.supervised.gaussian_nb.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
