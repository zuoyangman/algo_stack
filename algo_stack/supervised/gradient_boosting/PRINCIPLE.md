# Gradient Boosting — Principle

## 1. Problem statement

Build an additive model `F_M(x) = F_0 + η Σ_{m=1}^M h_m(x)` where each weak
learner `h_m` is a shallow regression tree fit to the negative gradient of a
loss w.r.t. the current prediction `F`.

## 2. Losses

**Regression (squared loss)**  
`L(y, F) = ½ (y − F)²` → negative gradient = residual `y − F`.  
`F_0 = mean(y)`.

**Binary classification (logistic / binomial deviance)**  
`L(y, F) = log(1 + e^F) − y F` with `y ∈ {0,1}`.  
Negative gradient = `y − σ(F)`.  
`F_0 = log(p̄ / (1 − p̄))` (log-odds of class prior).  
Class probability: `P(y=1 | x) = σ(F_M(x))`.

## 3. Algorithm

```
F ← F_0
for m = 1 .. M:
    r_i ← −∂L/∂F |_{F(x_i)}
    fit tree h_m to {(x_i, r_i)}
    F ← F + η · h_m(x)
```

## 4. Practical notes

- Small `learning_rate` + more trees usually generalises better.
- Shallow trees (`max_depth` 2–4) are the usual weak learners.
- This implementation uses plain MSE leaf means on the gradient; Newton
  leaf weighting is a common upgrade (see EXTENSION).
