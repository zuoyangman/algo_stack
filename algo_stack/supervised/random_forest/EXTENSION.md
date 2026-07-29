# Random Forest — Extension Guide

## 1. Extension surface

- **Base learner**: swap `DecisionTreeClassifier` / `Regressor` for extra
  trees (extra-trees style random thresholds).
- **Aggregation**: hard majority vote, weighted by OOB accuracy, stacking.
- **Sampling**: stratified bootstrap, subsample fraction `< 1.0`.

## 2. Common variants

### 2.1 Extra-Trees

At each split, draw random thresholds instead of scanning midpoints — lower
variance, sometimes higher bias.

### 2.2 Out-of-bag estimates

Track which indices were left out of each bootstrap and score OOB accuracy /
R² without a held-out set.

### 2.3 Feature importances

Accumulate impurity decrease per feature across trees (MDI) or permute
columns (MDA).

## 3. Testing checklist

- RF accuracy ≥ single deep tree on label-noisy blobs (or lower variance
  across seeds).
- `predict_proba` rows sum to 1.
- `n_estimators` matches `len(estimators_)`.
