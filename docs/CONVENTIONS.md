# Directory & Coding Conventions

Every algorithm in `algo_stack` follows the **same layout and same API
contract**. The goal is that a reader who has understood one module can predict
exactly where to find things in any other module.

---

## 1. Directory layout per algorithm

```
algo_stack/<category>/<algorithm_name>/
├── __init__.py            # re-exports the public estimator class(es)
├── README.md              # quick-start, public API, examples
├── PRINCIPLE.md           # math, derivation, complexity, design choices
├── EXTENSION.md           # how to extend / variants / pitfalls
├── <algorithm_name>.py    # the reference NumPy implementation
└── example.py             # runnable demo: `python -m algo_stack...example`
```

`<category>` is one of:

- `supervised`     – needs labelled targets `y`
- `unsupervised`   – uses only `X`
- `preprocessing`  – transformers that produce new `X`
- `metrics`        – pure functions (not estimators)
- `parsing`        – multimodal document ingestion (文 / 图 / 音 / 视)
- *(extend as needed)*

`<algorithm_name>` must be **lower_snake_case** and match the canonical name of
the algorithm (e.g. `linear_regression`, not `linreg`).

---

## 2. The three documentation files

### `README.md` — usage-facing
- One-line description.
- Quick install / import snippet.
- **Public API table** (class, constructor args, fitted attributes, methods).
- A short runnable usage example (copy-paste ready).
- A pointer to `PRINCIPLE.md` and `EXTENSION.md`.

### `PRINCIPLE.md` — theory-facing
- Problem statement (what does this algorithm solve?).
- Mathematical model + loss / objective.
- Derivation of the update rule / closed form.
- Computational complexity (time + memory) and numerical caveats.
- Notes on the specific design decisions made in `<algorithm_name>.py`.

### `EXTENSION.md` — developer-facing
- Which knobs are designed to be tweaked (constructor args, hooks).
- How to add a new variant (e.g. new regulariser, new kernel, new solver)
  *without* breaking the existing tests / API.
- Known pitfalls and invariants the implementation relies on.
- Suggested follow-up algorithms / extensions.

Use [`docs/templates/`](templates/) as starting points.

---

## 3. Coding conventions for the reference implementation

### Class API (scikit-learn-flavoured)

All estimators inherit from `algo_stack._base.BaseEstimator` and one of:

- `RegressorMixin`     – exposes `predict` and `score` (R² by default).
- `ClassifierMixin`    – exposes `predict`, `predict_proba` (if applicable),
  and `score` (accuracy by default).
- `ClusterMixin`       – exposes `predict` (cluster id) and `fit_predict`.
- `TransformerMixin`   – exposes `transform` and `fit_transform`.

The minimum contract for every supervised estimator is:

```python
class MyAlgo(BaseEstimator, RegressorMixin):
    def __init__(self, *, hyperparam1=..., ...):
        # ONLY store args — no work here.
        self.hyperparam1 = hyperparam1

    def fit(self, X, y):
        # learn, store fitted attrs with trailing underscore (e.g. self.coef_)
        return self

    def predict(self, X):
        # return predictions
        ...
```

### Style rules

- **NumPy only** in the core math. `scipy` is allowed only when explicitly
  documented (e.g. for `linalg.lstsq` fall-backs); never use `sklearn` inside
  the reference implementation.
- All public arrays are `np.ndarray` of `dtype=float64` after validation.
- Constructor arguments are **keyword-only** (use `*,` in the signature) so we
  never silently break callers when new hyperparameters are added.
- Fitted attributes end with `_` (e.g. `coef_`, `intercept_`, `cluster_centers_`).
- All randomness flows through a `random_state` (int or `np.random.Generator`)
  validated by `algo_stack.utils.validation.check_random_state`.
- Always validate inputs with `algo_stack.utils.validation.check_array` /
  `check_X_y` at the top of `fit`/`predict`.

### Tests

- One file per algorithm in `tests/test_<algorithm_name>.py`.
- Tests must run in **under one second** on a laptop — use small synthetic
  data, not real datasets.
- At minimum: (a) shape / API smoke test, (b) a numerical-correctness test
  against a closed-form answer or an easy-to-verify property.

---

## 4. Adding a new algorithm – checklist

1. `mkdir -p algo_stack/<category>/<algorithm_name>`
2. Copy the three `docs/templates/*.template.md` files and fill them in.
3. Implement `<algorithm_name>.py` following the API rules above.
4. Write `example.py` so that `python -m algo_stack.<category>.<name>.example`
   prints something informative.
5. Add `tests/test_<algorithm_name>.py`.
6. Re-export the class from the folder's `__init__.py`.
7. Update [`docs/ROADMAP.md`](ROADMAP.md) and the main `README.md` table.
