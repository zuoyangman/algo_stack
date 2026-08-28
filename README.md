# algo_stack

A growing, didactic collection of **from-scratch machine-learning / data-science
algorithm implementations** (Python + NumPy first, no scikit-learn shortcuts in
the core logic). The repository is designed to be:

- **Readable** – every algorithm ships with prose explaining *what* and *why*.
- **Extensible** – every algorithm ships with explicit guidance on *how to
  extend it* (variants, regularisation, alternative solvers, etc.).
- **Self-contained** – every algorithm lives in its own folder with code,
  docs, and a runnable example.
- **Testable** – every algorithm has a corresponding `pytest` test that
  validates correctness on a small synthetic problem.

---

## Repository layout

```
algo_stack/
├── README.md                <- this file
├── requirements.txt
├── pyproject.toml
├── docs/
│   ├── CONVENTIONS.md       <- directory & coding conventions every algorithm follows
│   ├── ROADMAP.md           <- checklist of algorithms (done + planned)
│   └── templates/           <- copy-paste templates for new algorithms
│       ├── README.template.md
│       ├── PRINCIPLE.template.md
│       └── EXTENSION.template.md
├── algo_stack/              <- the importable Python package
│   ├── __init__.py
│   ├── _base.py             <- BaseEstimator / mixin classes
│   ├── utils/               <- shared metrics / preprocessing / validation / activations / optim
│   ├── supervised/
│   │   ├── linear_regression/
│   │   ├── logistic_regression/
│   │   ├── knn/
│   │   ├── perceptron/
│   │   ├── softmax_classifier/
│   │   ├── mlp/
│   │   ├── cnn/
│   │   └── rnn/
│   └── unsupervised/
│       ├── kmeans/
│       └── autoencoder/
│   └── parsing/             <- 文 / 图 / 音 / 视 document parsing
└── tests/                   <- pytest suite, one file per algorithm
```

Every algorithm folder follows the **same four-file convention**
(see [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md)):

| File            | Purpose                                                      |
| --------------- | ------------------------------------------------------------ |
| `README.md`     | Quick start, public API, usage examples.                     |
| `PRINCIPLE.md`  | Math, derivations, complexity, design choices.               |
| `EXTENSION.md`  | How to plug in variants / regularisers / new solvers safely. |
| `*.py` + `example.py` | Reference NumPy implementation + runnable demo.        |

---

## Quick start

```bash
git clone <this-repo>
cd algo_stack
pip install -e .          # installs the algo_stack package + numpy
pip install -r requirements-dev.txt  # pytest etc. (optional)
pytest                    # run the test suite
```

Use any algorithm as a normal Python class:

```python
import numpy as np
from algo_stack.supervised.linear_regression import LinearRegression

X = np.random.randn(100, 3)
y = X @ np.array([1.5, -2.0, 0.5]) + 0.1 * np.random.randn(100)

model = LinearRegression().fit(X, y)
print(model.coef_, model.intercept_)
print("R^2 =", model.score(X, y))
```

Run any algorithm's standalone demo:

```bash
python -m algo_stack.supervised.linear_regression.example
```

---

## Currently implemented

| Category      | Algorithm            | Module path                                              |
| ------------- | -------------------- | -------------------------------------------------------- |
| Supervised    | Linear Regression    | `algo_stack.supervised.linear_regression`                |
| Supervised    | Logistic Regression  | `algo_stack.supervised.logistic_regression`              |
| Supervised    | K-Nearest Neighbours | `algo_stack.supervised.knn`                              |
| Supervised    | Perceptron           | `algo_stack.supervised.perceptron`                       |
| Supervised    | Softmax Classifier   | `algo_stack.supervised.softmax_classifier`               |
| Supervised    | Multi-Layer Perceptron | `algo_stack.supervised.mlp`                            |
| Supervised    | CNN (toy)            | `algo_stack.supervised.cnn`                              |
| Supervised    | RNN (vanilla, toy)   | `algo_stack.supervised.rnn`                              |
| Unsupervised  | K-Means              | `algo_stack.unsupervised.kmeans`                         |
| Unsupervised  | Autoencoder          | `algo_stack.unsupervised.autoencoder`                    |

Shared neural-network primitives: `algo_stack.utils.activations`, `algo_stack.utils.optim`.

### Document parsing — 文 / 图 / 音 / 视

| Component | Module |
| --------- | ------ |
| Unified parser | `algo_stack.parsing` (`parse`, `DocumentParser`) |
| 文 Text | `algo_stack.parsing.text` |
| 图 Image | `algo_stack.parsing.image` |
| 音 Audio | `algo_stack.parsing.audio` |
| 视 Video | `algo_stack.parsing.video` |

```bash
python -m algo_stack.parsing.example
```

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the planned algorithm list.

---

## Contributing a new algorithm

1. Read [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md).
2. Pick a slot under `algo_stack/<category>/<name>/`.
3. Copy the three doc templates from `docs/templates/` and fill them in.
4. Inherit from `algo_stack._base.BaseEstimator` (and one of the mixins).
5. Add a test in `tests/test_<name>.py`.
6. Update `docs/ROADMAP.md` and the *Currently implemented* table above.
