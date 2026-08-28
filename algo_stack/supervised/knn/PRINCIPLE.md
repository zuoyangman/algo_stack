# KNN — Principle

## 1. Problem statement

KNN is a **non-parametric, instance-based** learner: no model parameters are
estimated at fit time. Instead, predictions are made by looking up the `k`
closest training samples to each query.

- Classification: predict the (weighted) majority class among the `k`
  neighbours.
- Regression: predict the (weighted) mean of the `k` neighbour targets.

Underlying assumption: the target function is locally smooth — points close
in input space have similar labels.

## 2. Mathematical model

Let `D = {(x_i, y_i)}_{i=1..n}` be the training set and `d(·, ·)` a distance.
For a query `x`, define the index set of its `k`-nearest neighbours
`N_k(x) ⊂ {1,…,n}`. The prediction is

```
classification:  ŷ(x) = argmax_c Σ_{i ∈ N_k(x)} w_i · 1[y_i = c]
regression   :  ŷ(x) = (Σ_{i ∈ N_k(x)} w_i y_i) / (Σ_{i ∈ N_k(x)} w_i)
```

with `w_i = 1` for `weights="uniform"` and `w_i = 1 / (d(x, x_i) + ε)` for
`weights="distance"`.

## 3. Derivation

There is no training-time optimisation — KNN is a **memorisation method**.
The "learning" happens implicitly via the choice of `k`, distance metric,
and weighting. Theoretical justification: as `n → ∞` with `k → ∞` and
`k / n → 0`, the KNN classification error converges to twice the Bayes
error (Cover & Hart, 1967).

## 4. Algorithm

`fit(X, y)`:

```
1. Validate (X, y) and store them as X_train_, y_train_.
2. For classifier: integer-encode y via np.unique → y_train_ holds indices,
   classes_ holds the label vocabulary.
```

`predict(X_query)`:

```
1. Compute pairwise squared L2 distances between X_query and X_train_
   using ||a−b||² = ||a||² + ||b||² − 2 a·b  (one matrix multiply).
2. For each row, use np.argpartition(d², k-1) to get the k smallest
   indices in O(n_train) per query, then argsort just those k.
3. Aggregate the k neighbour labels (vote for classifier, mean for regressor),
   optionally weighting by 1/dist.
```

## 5. Complexity

| Quantity | Cost |
| -------- | ---- |
| Training time   | `O(1)` (just memorisation). |
| Memory          | `O(n d)` (store all training samples). |
| Prediction time | `O(n_train · d)` per query for the matrix multiply, plus `O(n_train + k log k)` for the partition+sort. |

## 6. Numerical & implementation notes

- Computing distances via `||a||² + ||b||² − 2 a·b` is much faster than a
  Python loop, but can return tiny **negative** values for very close points
  due to floating-point cancellation. We clip with `np.maximum(d², 0)` before
  taking the square root.
- `np.argpartition` is used instead of a full sort: `O(n)` vs. `O(n log n)`
  per query when `k ≪ n`. We then sort *only the k* selected indices so the
  "1st nearest", "2nd nearest", … ordering is correct.
- `predict_proba` for the classifier uses `np.add.at(...)` for a vectorised
  scatter-add of neighbour votes into the class-probability matrix.
- We add `1e-12` to distances before inverting them in `weights="distance"`
  so that exact duplicates do not cause a division by zero.
