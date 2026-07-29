# Decision Tree — Principle

## 1. Problem statement

A decision tree partitions the input space with recursive axis-aligned splits
so that each leaf is (approximately) pure w.r.t. the target. Classification
uses Gini impurity; regression uses mean squared error.

## 2. Mathematical model

At a node with samples `(X_S, y_S)`, candidate splits are pairs `(j, t)` that
send `x_j ≤ t` left and `x_j > t` right. The impurity decrease (gain) is

```
ΔI(j, t) = I(S) − (|S_L|/|S|) I(S_L) − (|S_R|/|S|) I(S_R)
```

- Classification: `I = Gini = 1 − Σ_c p_c²`
- Regression: `I = MSE = (1/|S|) Σ_i (y_i − ȳ)²`

Grow until purity, depth, or sample-count constraints stop further splits.
Leaf prediction: class frequencies (clf) or mean target (reg).

## 3. Algorithm

1. Recursively consider all features in a (possibly subsampled) feature set.
2. For each feature, evaluate midpoints between sorted unique values
   (subsample midpoints if there are many).
3. Pick the split with maximum gain; recurse on children.
4. At prediction time, walk the path from root to leaf.

## 4. Complexity

Let `n` samples, `d` features, depth `h`, and `T` candidate thresholds per
feature. Rough cost per node is `O(k · T · n)` for `k = max_features`, so full
growth is typically `O(k · T · n · h)` (much cheaper than exhaustive unique
value scans on dense continuous columns).

## 5. Practical notes

- XOR / checkerboard patterns need `max_depth ≥ 2`.
- Deep trees overfit; forests and boosting average / residual-correct them.
- `max_features < d` injects randomness used by Random Forests.
