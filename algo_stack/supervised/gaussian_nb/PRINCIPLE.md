# Gaussian Naive Bayes — Principle

## 1. Problem statement

Naive Bayes classifies by maximising `p(y|x) ∝ p(x|y) p(y)`. The "naive"
assumption is conditional independence of features given the class:

```
p(x | y = k) = Π_j p(x_j | y = k).
```

GaussianNB models each margin as `x_j | y=k ~ N(θ_{k,j}, σ²_{k,j})`.

## 2. Log joint likelihood

```
log p(x, y=k) = log π_k
              + Σ_j [ −½ log(2π σ²_{k,j}) − (x_j − θ_{k,j})² / (2 σ²_{k,j}) ]
```

Predict `ŷ = argmax_k log p(x, y=k)`. Softmax of the joints gives
`predict_proba`.

## 3. Parameter estimates

```
π̂_k       = n_k / n
θ̂_{k,j}   = mean of feature j in class k
σ̂²_{k,j}  = variance of feature j in class k  (+ var_smoothing floor)
```

## 4. Algorithm

```
1. For each class k: compute theta_[k], var_[k], counts.
2. Add epsilon = var_smoothing · max_feature_var to all variances.
3. class_prior_ ← counts/n (or user priors).
4. Predict via argmax of joint log-likelihood.
```

## 5. Complexity

| Quantity | Cost |
| -------- | ---- |
| Training | `O(n d)` |
| Prediction | `O(K d)` per sample |

## 6. Notes

- Independence is almost never true, yet NB often works surprisingly well.
- Near-zero variance features need `var_smoothing` to avoid `-inf` log probs.
