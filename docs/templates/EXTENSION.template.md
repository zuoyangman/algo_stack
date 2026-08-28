# <Algorithm Name> — Extension Guide

This document describes **how to extend the reference implementation safely**.
Treat it as the contract between the original author and future contributors.

## 1. Extension surface

These are the parts of the code that were explicitly designed to be modified:

- Constructor arguments: `<list of hyperparameters that can be added / tuned>`.
- Hooks / overridable methods: `<list of methods that subclasses may override>`.
- Pluggable utility functions: `<which `algo_stack.utils.*` functions can be
  swapped without changing the algorithm's correctness>`.

Anything **not** listed here is considered internal and may change without
notice.

## 2. Common variants and how to add them

For each likely variant, describe the minimal diff:

### 2.1 Adding a new regulariser / penalty / kernel / …

> Step-by-step: where to add the math, which method to override, which test to
> add. Refer to existing code by file:function name.

### 2.2 Swapping the solver / optimiser

> ...

### 2.3 Supporting a new data shape (sparse, batched, streaming)

> ...

## 3. Invariants the implementation relies on

These invariants must be preserved by any extension:

- `fit` returns `self` and does not mutate inputs.
- All fitted attributes end with `_`.
- `X` is validated to be a 2-D float array; `y` shape rules: ...
- ...

## 4. Pitfalls

- ...

## 5. Suggested follow-up algorithms

If you understood this module, the natural next algorithms to implement are:

- `<algorithm>` (link to roadmap entry)
- `<algorithm>` (link to roadmap entry)
