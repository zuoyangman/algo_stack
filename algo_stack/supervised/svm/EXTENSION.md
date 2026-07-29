# SVM — Extension Guide

## 1. Extension surface

- **Kernels**: add `"polynomial"` / custom callables via `algo_stack.utils.kernels.kernel_matrix`.
- **Solver**: replace `_smo_binary` with dual coordinate descent, LIBSVM-style
  working-set selection, or a quadratic-program backend.
- **Multiclass**: swap OvR for One-vs-One or Crammer–Singer.
- **SVR**: replace the Kernel Ridge solve with ε-SVR SMO (same KKT machinery
  as classification, with two sets of dual variables).

## 2. Adding ε-SVR SMO

The ε-insensitive dual introduces dual variables `α, α*` with
`0 ≤ α, α* ≤ C` and prediction
`f(x) = Σ_i (α_i − α*_i) K(x_i, x) + b`.
Reuse `_smo_binary`'s pair updates with the modified KKT conditions for the
tube loss; keep `SVC` unchanged.

## 3. Invariants

- `fit` does not mutate the caller's `X` / `y`.
- Binary `dual_coef_` equals `(α ⊙ y)` restricted to support vectors, so
  `decision_function` is `K(X, SV) @ dual_coef_ + intercept_`.
- `gamma_` is the resolved float used at fit time; prediction must use it,
  not re-resolve from the query matrix.

## 4. Pitfalls

- Building the full `n × n` Gram matrix limits `n` to a few thousand.
- Too-small `max_iter` leaves KKT violations — accuracy may still look fine
  on easy blobs but fail on harder sets.
- `C` and `gamma` interact strongly for RBF; default `"scale"` is a sane start.
- OvR scores are not calibrated probabilities.

## 5. Suggested follow-ups

- Nu-SVC / Nu-SVR
- Linear SVM via Pegasos / dual coordinate descent (no kernel matrix)
- Platt scaling / isotonic calibration for probabilities
- Sparse approximate kernels (Nyström, random Fourier features)
