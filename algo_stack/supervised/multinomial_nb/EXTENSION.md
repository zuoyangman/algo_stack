# Multinomial Naive Bayes — Extension Guide

## 1. Extension surface

- **`alpha`**: Lidstone smoothing; try values in `{0, 0.01, 0.1, 1}`.
- **`fit_prior` / `class_prior`**: control class imbalance handling.
- **`partial_fit`**: accumulate `feature_count_` / `class_count_` across
  mini-batches for streaming text.

## 2. Common variants

### 2.1 Complement NB

Estimate weights from the complement of each class — often better on
skewed text corpora.

### 2.2 TF-IDF front-end

Pipe documents through a vectoriser + TF-IDF transformer before
`MultinomialNB` (NB still expects non-negative inputs).

### 2.3 Sparse matrices

Accept `scipy.sparse` and use sparse `X @ feature_log_prob_.T` for large
vocabularies (keep this module NumPy-dense for didactic clarity).

## 3. Invariants

- `feature_log_prob_.shape == (n_classes, n_features)`
- `exp(feature_log_prob_).sum(axis=1) ≈ 1`
- Inputs to `fit` / `predict` are non-negative

## 4. Pitfalls

- Feeding standardised (possibly negative) features → ValueError.
- Extremely sparse rare tokens: raise `alpha` or prune the vocabulary.
