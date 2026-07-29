# Multinomial Naive Bayes

> Naive Bayes with a multinomial event model — the classic baseline for text
> classification and other non-negative count / TF feature vectors. Prediction
> is entirely in log-space.

## Install / import

```python
from algo_stack.supervised.multinomial_nb import MultinomialNB
```

## Public API

| Name | Type | Description |
| ---- | ---- | ----------- |
| `MultinomialNB(*, alpha=1.0, fit_prior=True, class_prior=None)` | constructor | `alpha` = Laplace smoothing. |
| `.fit(X, y)` | method | Estimate smoothed feature log-probs. |
| `.predict(X)` | method | MAP class labels (log-space). |
| `.predict_proba(X)` / `.predict_log_proba(X)` | method | Posteriors. |
| `.score(X, y)` | method | Accuracy. |
| `.feature_log_prob_` | fitted | `log P(j \| k)`, shape `(n_classes, n_features)`. |
| `.class_log_prior_` | fitted | `log π_k`. |

## Quick start

```python
import numpy as np
from algo_stack.supervised.multinomial_nb import MultinomialNB

X = np.array([[3, 0, 1], [2, 0, 0], [0, 3, 1], [0, 2, 2]], dtype=float)
y = np.array([0, 0, 1, 1])
clf = MultinomialNB(alpha=1.0).fit(X, y)
print(clf.predict(X), clf.score(X, y))
```

## Run the bundled example

```bash
python -m algo_stack.supervised.multinomial_nb.example
```

## Where to go next

- Theory & derivation → [`PRINCIPLE.md`](PRINCIPLE.md)
- Extending / customising → [`EXTENSION.md`](EXTENSION.md)
