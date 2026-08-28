# Perceptron — Principle

## 1. Problem statement

Learn a linear decision boundary that separates two (or more) classes. The
Perceptron is the simplest neural model: a single neuron with a step activation
`ŷ = sign(w·x + b)`.

**Convergence guarantee** (Perceptron Convergence Theorem): if the data are
linearly separable with margin `γ > 0`, the algorithm finds a separating
hyperplane in a finite number of updates.

## 2. Mathematical model

Binary case with labels `y ∈ {-1, +1}`:

```
f(x) = sign(w·x + b)
```

Update on a misclassified sample `(x_i, y_i)`:

```
w ← w + η · y_i · x_i
b ← b + η · y_i        (if fit_intercept)
```

Multiclass uses **one-vs-rest**: maintain one weight vector per class; on a
mistake between class `k` (true) and class `j` (predicted):

```
w_k ← w_k + η · x_i
w_j ← w_j − η · x_i
```

## 3. Derivation

The update moves the hyperplane toward the misclassified point: adding
`y_i · x_i` to `w` increases the score `w·x_i` in the direction of the
correct label.

## 4. Algorithm

```
Initialise w = 0, b = 0
for epoch = 1 .. n_iter:
    shuffle training indices
    mistakes = 0
    for each (x_i, y_i):
        if sign(w·x_i + b) ≠ y_i:
            w ← w + η · y_i · x_i;  b ← b + η · y_i
            mistakes += 1
    if mistakes / n ≤ tol: stop
```

## 5. Complexity

| Quantity | Cost |
| -------- | ---- |
| Per epoch | `O(n · d)` |
| Memory    | `O(d)` (or `O(K · d)` for OvR multiclass) |

## 6. Numerical & implementation notes

- Labels are mapped to `{-1, +1}` internally for the binary case; `predict`
  maps back to the original `classes_` vocabulary.
- The Perceptron **cannot** solve XOR or any non-linearly-separable problem —
  that limitation motivated multi-layer networks (see `supervised/mlp`).
- `learning_rate` only scales convergence speed for separable data; it does
  not change the final separating hyperplane (up to scaling).
