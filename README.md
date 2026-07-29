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
├── README.md
├── docs/CONVENTIONS.md · ROADMAP.md · templates/
├── algo_stack/
│   ├── _base.py · utils/ (validation, metrics, preprocessing, activations, optim, kernels)
│   ├── supervised/     # regressors & classifiers
│   ├── unsupervised/   # clustering, DR, density, autoencoders
│   ├── preprocessing/  # scalers, encoders, feature expansion
│   └── metrics/        # curated scoring functions
└── tests/
```

Every algorithm folder follows the **same four-file convention**
(see [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md)):

| File | Purpose |
| ---- | ------- |
| `README.md` | Quick start, public API, usage examples |
| `PRINCIPLE.md` | Math, derivations, complexity, design choices |
| `EXTENSION.md` | How to plug in variants safely |
| `*.py` + `example.py` | Reference NumPy implementation + runnable demo |

---

## Quick start

```bash
git clone <this-repo>
cd algo_stack
pip install -e .
pip install -r requirements-dev.txt
pytest
```

```python
import numpy as np
from algo_stack.supervised.linear_regression import LinearRegression

X = np.random.randn(100, 3)
y = X @ np.array([1.5, -2.0, 0.5]) + 0.1 * np.random.randn(100)
model = LinearRegression().fit(X, y)
print(model.coef_, model.intercept_, model.score(X, y))
```

```bash
python -m algo_stack.supervised.random_forest.example
```

---

## Currently implemented

### Supervised — linear / GLM
| Algorithm | Module |
| --------- | ------ |
| Linear Regression | `algo_stack.supervised.linear_regression` |
| Ridge | `algo_stack.supervised.ridge` |
| Lasso | `algo_stack.supervised.lasso` |
| Logistic Regression | `algo_stack.supervised.logistic_regression` |
| Softmax Classifier | `algo_stack.supervised.softmax_classifier` |
| Perceptron | `algo_stack.supervised.perceptron` |
| Linear Discriminant Analysis | `algo_stack.supervised.lda` |
| Bayesian Linear Regression | `algo_stack.supervised.bayesian_linear_regression` |

### Supervised — instance / tree / probabilistic / kernel
| Algorithm | Module |
| --------- | ------ |
| K-Nearest Neighbours | `algo_stack.supervised.knn` |
| Decision Tree (CART) | `algo_stack.supervised.decision_tree` |
| Random Forest | `algo_stack.supervised.random_forest` |
| Gradient Boosting | `algo_stack.supervised.gradient_boosting` |
| Gaussian Naive Bayes | `algo_stack.supervised.gaussian_nb` |
| Multinomial Naive Bayes | `algo_stack.supervised.multinomial_nb` |
| SVM (SVC / SVR) | `algo_stack.supervised.svm` |
| Kernel Ridge | `algo_stack.supervised.kernel_ridge` |

### Supervised — neural networks
| Algorithm | Module |
| --------- | ------ |
| Multi-Layer Perceptron | `algo_stack.supervised.mlp` |
| CNN (toy) | `algo_stack.supervised.cnn` |
| RNN (vanilla) | `algo_stack.supervised.rnn` |
| LSTM / GRU | `algo_stack.supervised.lstm` |

### Unsupervised
| Algorithm | Module |
| --------- | ------ |
| K-Means | `algo_stack.unsupervised.kmeans` |
| Mini-batch K-Means | `algo_stack.unsupervised.minibatch_kmeans` |
| DBSCAN | `algo_stack.unsupervised.dbscan` |
| Gaussian Mixture (EM) | `algo_stack.unsupervised.gmm` |
| Agglomerative Clustering | `algo_stack.unsupervised.hierarchical` |
| PCA | `algo_stack.unsupervised.pca` |
| Kernel PCA | `algo_stack.unsupervised.kernel_pca` |
| t-SNE | `algo_stack.unsupervised.tsne` |
| UMAP (simplified) | `algo_stack.unsupervised.umap` |
| Kernel Density Estimation | `algo_stack.unsupervised.kde` |
| Autoencoder | `algo_stack.unsupervised.autoencoder` |
| Denoising Autoencoder | `algo_stack.unsupervised.denoising_autoencoder` |
| Variational Autoencoder | `algo_stack.unsupervised.vae` |
| Convolutional Autoencoder | `algo_stack.unsupervised.conv_autoencoder` |

### Preprocessing & metrics
| Algorithm | Module |
| --------- | ------ |
| StandardScaler | `algo_stack.preprocessing.standard_scaler` |
| MinMaxScaler | `algo_stack.preprocessing.minmax_scaler` |
| OneHotEncoder | `algo_stack.preprocessing.one_hot_encoder` |
| PolynomialFeatures | `algo_stack.preprocessing.polynomial_features` |
| Metrics suite | `algo_stack.metrics` |
| L-BFGS + Wolfe line search | `algo_stack.optimization.lbfgs` |

Shared primitives: `algo_stack.utils.activations`, `optim`, `kernels`,
`neighbors` (KD-Tree / Ball-Tree), `lbfgs` (also under `algo_stack.optimization.lbfgs`).

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the full checklist.

---

## Contributing a new algorithm

1. Read [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md).
2. Pick a slot under `algo_stack/<category>/<name>/`.
3. Copy the three doc templates from `docs/templates/` and fill them in.
4. Inherit from `algo_stack._base.BaseEstimator` (and the relevant mixin).
5. Add a test in `tests/test_<name>.py`.
6. Update `docs/ROADMAP.md` and the tables above.
