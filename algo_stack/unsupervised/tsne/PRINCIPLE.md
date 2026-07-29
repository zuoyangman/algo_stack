# t-SNE — Principle

## 1. Problem statement

Map high-d points to low-d so that neighbourhood probabilities match, minimising KL`(P‖Q)`.

## 2. Model

High-d: binary-search Gaussian bandwidths for target perplexity; symmetrise `P`.
Low-d: Student-t affinities `Q`. Gradient descent with early exaggeration + momentum.

## 3. Algorithm

```
build conditional P_i|j via binary search on β; symmetrise
init Y ~ N(0, 1e-4)
for t = 1..n_iter:
  compute Q from Student-t
  gradient of KL; adaptive gains + momentum update
  exaggerate P for first 250 iters
```

## 4. Complexity

`O(n²)` per iteration — intended for `n ≲ 200`.

## 5. Notes

`transform` on new data is unsupported (transductive method).
