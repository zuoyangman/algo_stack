# Gaussian Naive Bayes — Extension Guide

## 1. Extension surface

- **`var_smoothing` / `priors`**: already exposed.
- **Online updates**: maintain running mean/variance (Welford) for streaming
  `partial_fit`.
- **Tied variance**: share one `var_` across classes (closer to LDA under
  independence).

## 2. Common variants

### 2.1 Bernoulli / Multinomial NB

Different likelihoods for binary / count features — see sibling modules.

### 2.2 Complement NB

Use complement-class statistics for imbalanced text data.

### 2.3 Kernel Density NB

Replace the Gaussian margin with a 1-D KDE per feature/class.

## 3. Invariants

- `theta_.shape == var_.shape == (n_classes, n_features)`
- `class_prior_.sum() ≈ 1`
- `predict` returns labels from `classes_`

## 4. Pitfalls

- Highly correlated features violate independence → consider LDA instead.
- Discrete / sparse count data → use MultinomialNB, not GaussianNB.
