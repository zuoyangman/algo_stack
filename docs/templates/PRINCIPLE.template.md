# <Algorithm Name> — Principle

## 1. Problem statement

State, in one paragraph, the problem being solved and the assumptions made
(e.g. "Given i.i.d. samples (x_i, y_i), assume y_i = w·x_i + b + ε with
ε ~ N(0, σ²); find w, b that maximise the likelihood…").

## 2. Mathematical model

Define notation, the hypothesis class, and the loss / objective.

```
L(θ) = ...
```

## 3. Derivation

Show how the update rule / closed form is derived from §2. Include the key
equations and any tricks used (e.g. adding a bias column, using the
log-sum-exp trick, normal equations, KKT conditions, …).

## 4. Algorithm

Pseudocode of the procedure that `fit` implements:

```
1. validate inputs
2. ...
3. ...
```

## 5. Complexity

| Quantity | Cost |
| -------- | ---- |
| Training time   | `O(...)` |
| Training memory | `O(...)` |
| Prediction time | `O(...) per sample` |

## 6. Numerical & implementation notes

Explain non-obvious choices in the code (e.g. why we use `lstsq` instead of
inverting `XᵀX`, why we centre features, log-sum-exp for softmax, etc.).
