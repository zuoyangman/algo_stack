# Kernel Density Estimation — Principle

## 1. Problem statement

Estimate `p(x)` from samples without assuming a parametric family.

## 2. Model

`p̂(x) = (1/(n h^d)) Σ_i K((x−x_i)/h)` with standard normal `K`.

## 3. Algorithm

```
store X_fit
score_samples: pairwise sq distances → log Gaussian kernels → logsumexp − log n
```

## 4. Complexity

`O(n_query · n_train · d)` per scoring call.

## 5. Notes

Log-domain evaluation avoids underflow for moderate `d`.
