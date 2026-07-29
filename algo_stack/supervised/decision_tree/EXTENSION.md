# Decision Tree — Extension Guide

## 1. Extension surface

- **Split criterion**: swap `_gini` / `_mse` (or add entropy, MAE, Poisson).
- **Threshold search**: `_candidate_thresholds` — replace with histograms,
  quantiles, or exact sorted-scan CART.
- **Tree structure**: `Node` dataclass; serialise to nested dicts or flatten
  to arrays for vectorised predict.
- **`max_features`**: `_resolve_max_features` already accepts `"sqrt"` /
  `"log2"` / fraction for bagging variants.

## 2. Common variants

### 2.1 Cost-complexity pruning

Grow a full tree, then prune by minimizing `R_α(T) = R(T) + α |T|` on a
validation set (Breiman CCP). Attach `ccp_alpha` and a post-fit prune pass.

### 2.2 Missing values

Surrogate splits or send missing values to the majority child. Hook into
`_predict_tree_row` and the split loop.

### 2.3 Oblique / multiway splits

Replace axis-aligned `(j, t)` with linear combinations or categorical
multiway partitions; keep the same recursive `Node` API.

## 3. Testing checklist

- XOR with `max_depth=1` fails, `max_depth=2` succeeds.
- Separable blobs → near-perfect train accuracy.
- Regressor recovers a piecewise-constant target within leaf noise.
