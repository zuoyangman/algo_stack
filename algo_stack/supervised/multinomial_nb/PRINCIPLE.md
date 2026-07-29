# Multinomial Naive Bayes — Principle

## 1. Problem statement

For documents (or any bag-of-tokens vector) `x ∈ ℕᵈ`, the multinomial event
model says

```
p(x | y = k) ∝ Π_j θ_{k,j}^{x_j},   Σ_j θ_{k,j} = 1,  θ_{k,j} ≥ 0.
```

Combined with a class prior `π_k`, MAP classification is

```
ŷ = argmax_k [ log π_k + Σ_j x_j log θ_{k,j} ].
```

## 2. Laplace smoothing

Maximum-likelihood `θ̂_{k,j} = N_{k,j} / N_k` assigns zero probability to
unseen tokens. Add-α smoothing:

```
θ̂_{k,j} = (N_{k,j} + α) / (N_k + α d),   α ≥ 0.
```

`α = 1` is classical Laplace smoothing.

## 3. Algorithm

```
1. Reject negative feature values.
2. feature_count_[k] ← Σ_{i:y_i=k} x_i
3. feature_log_prob_[k] ← log(count + α) − log(Σ (count + α))
4. class_log_prior_ ← log(n_k / n)   (or uniform / user-supplied)
5. Predict: argmax_k  x · feature_log_prob_[k] + class_log_prior_[k]
```

All steps after counting stay in log-space — never exponentiate `θ` until
`predict_proba`.

## 4. Complexity

| Quantity | Cost |
| -------- | ---- |
| Training | `O(n d)` |
| Prediction | `O(K d)` per sample (dense); sparse mat-vec is better for text |

## 5. Notes

- Features must be non-negative (counts or TF / TF-IDF work).
- For binary occurrence features prefer BernoulliNB.
