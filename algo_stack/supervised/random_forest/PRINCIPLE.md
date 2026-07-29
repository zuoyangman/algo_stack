# Random Forest — Principle

## 1. Problem statement

A single deep tree has low bias but high variance. Random Forests reduce
variance by averaging many decorrelated trees.

## 2. Mathematical model

For `b = 1..B`:

1. Draw a bootstrap sample `(X^{(b)}, y^{(b)})` of size `n` with replacement.
2. Grow a CART tree `T_b` with random feature subsets of size `max_features`
   at every split.

Predictions:

```
classification:  ŷ(x) = argmax_c (1/B) Σ_b P_b(c | x)
regression   :  ŷ(x) = (1/B) Σ_b T_b(x)
```

Decorrelation comes from bootstrap + feature bagging; averaging shrinks
variance roughly as `ρ·σ² + (1−ρ)·σ²/B` where `ρ` is inter-tree correlation.

## 3. Algorithm sketch

Loop `n_estimators` times: sample indices → fit `DecisionTree*` with
`max_features` and a fresh `random_state` → store in `estimators_`.

## 4. Practical notes

- Default `max_features="sqrt"` for classification-style bagging.
- Deeper trees are fine; overfit of individuals is averaged out.
- On noisy labels, RF is typically more stable than one deep tree.
